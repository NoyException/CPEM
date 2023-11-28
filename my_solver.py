import os
import torch
import glob
import cv2
from skimage import draw
from scipy.io import savemat, loadmat
import numpy as np

from core.data_utils import Preprocess, similar_transform_crop_to_origin, similar_transform_3d, overlying_image_resize
from model.resnet import ResNet50_3DMM
from model.mobilenetv2 import MobilenetV2_3DMM
from core.face_decoder import FaceDecoder

from utils import write_obj


class MySolver(object):

    def __init__(self, opts):
        self.opts = opts
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # face decoder
        self.FaceDecoder = FaceDecoder(opts.gpmm_model_path, opts.gpmm_delta_bs_path, batch_size=self.opts.batch_size, device=self.device, img_size=opts.input_size)

        # build model
        self.init_model()

        # load trained model
        self.restore_model(self.opts.test_iter)


    def init_model(self):
        print('Network Type: ', self.opts.network_type)
        if self.opts.network_type == 'ResNet50':
            self.network = ResNet50_3DMM(n_id=self.FaceDecoder.facemodel.n_id_para,
                                         n_bs=self.FaceDecoder.facemodel.n_bs_para,
                                         n_tex=self.FaceDecoder.facemodel.n_tex_para,
                                         n_rot=self.FaceDecoder.facemodel.n_rot_para,
                                         n_light=self.FaceDecoder.facemodel.n_light_para,
                                         n_tran=self.FaceDecoder.facemodel.n_tran_para)
        elif self.opts.network_type == 'mobilenet-v2':
            self.network = MobilenetV2_3DMM(n_id=self.FaceDecoder.facemodel.n_id_para,
                                            n_bs=self.FaceDecoder.facemodel.n_bs_para,
                                            n_tex=self.FaceDecoder.facemodel.n_tex_para,
                                            n_rot=self.FaceDecoder.facemodel.n_rot_para,
                                            n_light=self.FaceDecoder.facemodel.n_light_para,
                                            n_tran=self.FaceDecoder.facemodel.n_tran_para)

        self.network.to(self.device)


    def restore_model(self, resume_iters):
        """
        Restore the trained generators and discriminator.
        """
        if self.opts.checkpoint_path is not None and os.path.exists(self.opts.checkpoint_path):
            print('Loading the trained models from {}...'.format(self.opts.checkpoint_path))
            self.network.load_state_dict(torch.load(self.opts.checkpoint_path, map_location=lambda storage, loc: storage))
        else:
            print('Loading the trained models from step {}...'.format(resume_iters))
            net_path = os.path.join(self.opts.checkpoint_dir, '{}-network.ckpt'.format(resume_iters))
            if os.path.exists(net_path):
                self.network.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))
            else:
                assert False, 'Checkpoint flie not exist!'


    def infer_from_image_paths(self, frame, face_detector):
        # detect bbox
        if self.opts.detect_type == 'box':
            boxes = face_detector(frame)
            if len(boxes) == 0:
                print('No face detected')
                return
            bbox = boxes[0]
        else:
            bbox, _ = face_detector.detect(frame)
            if bbox is None:
                print('No face detected')
                return
        # preprocess data
        # [bs, c, h, w]
        input_data, roi_box = Preprocess(frame, self.opts.input_size, bbox, self.opts.detect_type, return_box=True)
        input_data = input_data.to(self.device)
        # network inference
        pred_coeffs = self.network(input_data)  # [bs, 159]

        # rendered_img: [1, h, w, 4]
        rendered_img, pred_lm2d, coeffs, mesh = self.FaceDecoder.decode_face(pred_coeffs, return_coeffs=True)
        render_mask = rendered_img[:, :, :, 3].detach()
        rendered_img = rendered_img[:, :, :, :3]

        # overlay processed image
        eval_intput_data = input_data * 255  # [0,1] -> [0,255]
        eval_intput_data = eval_intput_data.permute(0, 2, 3, 1)  # [bs, c, h, w] -> [bs, h, w, c]

        eval_mask = (render_mask > 0).type(torch.uint8)
        eval_mask = eval_mask.view(eval_mask.size(0), eval_mask.size(1), eval_mask.size(2), 1)
        eval_mask = eval_mask.repeat(1, 1, 1, 3)
        eval_overlay_images = rendered_img * eval_mask + eval_intput_data * (1 - eval_mask)
        eval_overlay_images = eval_overlay_images.cpu().numpy()
        eval_overlay_image = np.squeeze(eval_overlay_images)[:, :, ::-1]  # [h,w,c] BGR

        # overlay original image
        rendered_img = rendered_img.squeeze().cpu().numpy()
        rendered_mask = eval_mask.squeeze().cpu().numpy()
        rendered_img = rendered_img[:, :, ::-1]  # [h,w,c] BGR
        raw_img = frame.copy()
        composed_img = overlying_image_resize(roi_box, raw_img, rendered_img, rendered_mask)
        return composed_img

    def render_shape(self, image, face_detector):
        # if self.opts.save_path is None:
        #     save_path = os.path.join(self.opts.result_root, 'results_{}'.format(self.opts.test_iter))
        # else:
        #     save_path = self.opts.save_path
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        self.network.eval()
        with torch.no_grad():

            # detect bbox
            if self.opts.detect_type == 'box':
                boxes = face_detector(image)
                if len(boxes) == 0:
                    print('No face detected')
                    return
                bbox = boxes[0]
            else:
                bbox, _ = face_detector.detect(image)
                if bbox is None:
                    print('No face detected')
                    return
            # preprocess data
            # [bs, c, h, w]
            input_data, roi_box = Preprocess(image, self.opts.input_size, bbox, self.opts.detect_type, return_box=True)
            input_data = input_data.to(self.device)
            # network inference
            pred_coeffs = self.network(input_data)  # [bs, 159]

            # get vertices on the original image plane
            face_shape_2d = self.FaceDecoder.get_face_on_2d_plane(pred_coeffs)
            face_shape_2d = face_shape_2d[0].detach().cpu()
            face_shape_2d_ori = similar_transform_3d(face_shape_2d, roi_box, self.opts.input_size)

            # vertices input for render function
            face_shape_2d_ori_array = face_shape_2d_ori.numpy()
            face_shape_2d_ori_array = face_shape_2d_ori_array
            face_shape_2d_ori_array[:, 2] = -1 * face_shape_2d_ori_array[:, 2]
            face_shape_2d_ori_array = face_shape_2d_ori_array.astype(np.float32).copy(order='C')

            # triangle input for render function
            tri = self.FaceDecoder.facemodel.tri.detach().cpu() - 1
            triangle_array = tri.numpy().astype(np.int32)
            triangle_array = triangle_array[:, ::-1]
            triangle_array = triangle_array.copy(order='C')

            from render_api import render
            eval_overlay_image = image.copy()
            overlay_image = render(eval_overlay_image, face_shape_2d_ori_array, triangle_array, alpha=1.0)
            return overlay_image


    def run_facial_motion_retargeting(self, src_coeff_path, tgt_img, face_detector):
        '''
        src_coeff_path: source 3DMM parameter path or dirs with *.mat format
        target_img_path: retarget object
        '''
        # if self.opts.save_path is None:
        #     save_path = os.path.join(self.opts.result_root, 'results_{}'.format(self.opts.test_iter))
        # else:
        #     save_path = self.opts.save_path
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        self.network.eval()
        with torch.no_grad():
            # detect bbox
            if self.opts.detect_type == 'box':
                tgt_boxes = face_detector(tgt_img)
                if len(tgt_boxes) == 0:
                    print('No face detected')
                    return
                tgt_bbox = tgt_boxes[0]

            else:
                tgt_bbox, _ = face_detector.detect(tgt_img)
                if tgt_bbox is None:
                    print('No face detected')
                    return

            # preprocess data
            # [bs, c, h, w]
            tgt_input_data, tgt_roi_box = Preprocess(tgt_img, self.opts.input_size, tgt_bbox, self.opts.detect_type, return_box=True)
            tgt_input_data = tgt_input_data.to(self.device)

            # network inference
            tgt_pred_coeffs = self.network(tgt_input_data)  # [bs, 159]

            # file dirs or file path
            if os.path.isfile(src_coeff_path):
                src_coeff_path_list = [src_coeff_path]
            else:
                src_coeff_path_list = glob.glob(os.path.join(src_coeff_path, '*.mat'))
                src_coeff_path_list.sort()

            length = len(src_coeff_path_list)
            for idx, coeff_path in enumerate(src_coeff_path_list):
                print('process frames [{}/{}].'.format(idx+1, length))
                # fetch retargeting parameters
                src_coeffs = loadmat(coeff_path)
                src_bs_coeff = src_coeffs['coeff_bs']
                src_angles = src_coeffs['coeff_angle']
                src_translation = src_coeffs['coeff_translation']
                src_bs_coeff = torch.from_numpy(src_bs_coeff).to(self.device)
                src_angles = torch.from_numpy(src_angles).to(self.device)
                src_translation = torch.from_numpy(src_translation).to(self.device)

                # fetch identity parameters from source image
                tgt_id_coeff, tgt_bs_coeff, tgt_tex_coeff, tgt_angles, tgt_gamma, tgt_translation = self.FaceDecoder.Split_coeff(tgt_pred_coeffs)

                # get retargeting result
                combined_coeffs = torch.cat([tgt_id_coeff, src_bs_coeff, tgt_tex_coeff, src_angles, tgt_gamma, src_translation], dim=1)
                rendered_img, _, _, retargeted_mesh = self.FaceDecoder.decode_face(combined_coeffs, return_coeffs=True)
                rendered_img = rendered_img[:, :, :, :3]

                rendered_img = rendered_img.squeeze().cpu().numpy()
                rendered_img = rendered_img[:, :, ::-1]  # [h,w,c] BGR

                return rendered_img
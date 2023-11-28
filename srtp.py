import os
import socket
import struct
from io import BytesIO

import cv2
import numpy as np

from FaceBoxes import FaceBoxes  # face box detector
from my_solver import MySolver
from utils import set_seed

# set opts
dic = {
    "seed": 2408,
    "use_gpu": True,
    "log_dir": "logs/",
    "checkpoint_dir": "models/",
    "sample_dir": "samples/",
    "result_root": "results/",
    "checkpoint_path": "./data/pretrained_model/resnet50-id-exp-300000.ckpt",
    "gpmm_model_path": "data/BFM/BFM_model_front.mat",
    "gpmm_delta_bs_path": "data/BFM/mean_delta_blendshape.npy",
    "input_size": 224,
    "input_channel": 3,
    "n_frames": 4,
    "batch_size": 1,
    "conv_dim": 32,
    "network_type": "ResNet50",
    "mode": "demo",
    "test_iter": 300000,
    "image_path": None,
    "save_path": None,
    "onnx": False,
    "detect_type": "box",
    "save_mesh": False,
    "source_coeff_path": None,
    "target_image_path": None,
}


class Opts(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


opts = Opts(**dic)

set_seed(opts.seed)

solver = MySolver(opts)

if opts.onnx:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'

    from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX

    face_detector = FaceBoxes_ONNX()
else:
    face_detector = FaceBoxes()


def process_image(img):
    # return solver.infer_from_image_paths(img, face_detector)
    return solver.render_shape(img, face_detector)
    # return solver.run_facial_motion_retargeting("./data/demo_save/coeffs", img, face_detector)


# 创建socket对象，绑定端口，监听连接
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 8887))
server.listen(1)

print('Waiting for connection...')
while True:
    # 接受来自java客户端的连接
    client, address = server.accept()
    print('Connected to:', address)
    # 获取输入输出流
    in_stream = client.makefile(mode='rb')
    out_stream = client.makefile(mode='wb')
    # 接收来自java客户端的图片字节数组长度和内容
    length = struct.unpack('>I', in_stream.read(4))[0]
    image_bytes = in_stream.read(length)
    # 将字节数组转换为图片
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    # 对图片进行处理
    result_image = process_image(image)
    if result_image is None:
        # 写入0表示处理失败
        print("处理失败")
        out_stream.write(struct.pack('>I', 0))
        in_stream.close()
        out_stream.close()
        client.close()
        continue
    print("处理成功")
    # 将图片转换为字节数组
    result_bytes = BytesIO()
    _, buffer = cv2.imencode('.jpg', result_image)
    result_bytes.write(buffer)
    result_bytes = result_bytes.getvalue()
    # 发送处理后的图片字节数组长度和内容到java客户端
    out_stream.write(struct.pack('>I', len(result_bytes)))
    out_stream.write(result_bytes)
    out_stream.flush()
    # 关闭资源
    in_stream.close()
    out_stream.close()
    client.close()
server.close()

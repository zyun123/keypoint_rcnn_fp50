import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import torch
import cv2

model = onnx.load("deploy/keypoint_rcnn_r50_fpn.onnx")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
engine = backend.prepare(model,device)
input_data = cv2.imread("deploy/test.jpg").transpose(2, 0, 1)[None,:].astype(np.float32)/255
output_data = engine.run(input_data)
print(output_data)
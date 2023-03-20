import cv2
import onnxruntime as ort
import onnx
import numpy as np
import time
from network_files import KeypointRCNN
from backbone import resnet50_fpn_backbone
import torch
import os
import torchvision.models.detection as detection

onnx_path = "deploy/keypoint_rcnn_r50_fpn.onnx"
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

onnx_session = ort.InferenceSession(onnx_path,providers=['CUDAExecutionProvider','TensorrtExecutionProvider','CPUExecutionProvider'])

print(onnx_session.get_providers())
input = [node.name for node in onnx_session.get_inputs()]
print("input_name**:",input)

print("-----------------------------------------------------------------------------------")
output = [node.name for node in onnx_session.get_outputs()]
print("output_name**:",output)

input_name = input[0]
output_name = output[0]


'''
法一
image = cv2.imread("./test.jpg").transpose(2, 0, 1)
image = np.expand_dims(image,axis =0)
'''
image = cv2.imread("deploy/test.jpg").transpose(2, 0, 1)[None,:].astype(np.float32)/255

time1 = time.time()
ort_outs = onnx_session.run([output_name],{input_name:image})
time2 = time.time()
print("onnxruntime used time:#####",(time2-time1))
# print(ort_outs)




#############################normal predict
# num_classes = 1  # 不包含背景
# box_thresh = 0.5
# num_keypoints = 50
# weights_path = "./save_weights/model_19.pth"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def create_model(num_classes, box_thresh=0.5,num_keypoints=50):
#     backbone = resnet50_fpn_backbone()
#     model = KeypointRCNN(backbone,
#                      num_classes=num_classes,
#                      rpn_score_thresh=box_thresh,
#                      box_score_thresh=box_thresh,
#                      num_keypoints = num_keypoints)

#     return model
# model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh,num_keypoints = num_keypoints)
# # model = detection.keypointrcnn_resnet50_fpn()
# assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
# weights_dict = torch.load(weights_path, map_location='cpu')
# weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
# model.load_state_dict(weights_dict)
# # model.to(device)
# model.eval()
# img_height, img_width = (720,1280)
# init_img = torch.zeros((1, 3, img_height, img_width), device="cpu")
# model(init_img)
# predictions = model(torch.tensor(image))[0]
# def to_numpy(tensor):
#             if tensor.requires_grad:
#                 return tensor.detach().cpu().numpy()
#             else:
#                 return tensor.cpu().numpy()
# outputs,_ = torch.jit._flatten(predictions)
# outputs = list(map(to_numpy,outputs))

# for i in range(0,len(outputs)):
#     try:
#         torch.testing.assert_allclose(outputs[i],ort_outs[i],rtol = 1e-03,atol = 1e-05)
#         print("**************ok********************")
#     except AssertionError as error:
#         assert "(0.00%)" in str(error), str(error)


import onnx
from network_files import KeypointRCNN
from backbone import resnet50_fpn_backbone
import torch
import os
import torchvision.models.detection as detection
# detection.keypointrcnn_resnet50_fpn()
import cv2
num_classes = 1  # 不包含背景
box_thresh = 0.5
num_keypoints = 50
weights_path = "./save_weights/model_19.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_model(num_classes, box_thresh=0.5,num_keypoints=50):
    backbone = resnet50_fpn_backbone()
    model = KeypointRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh,
                     num_keypoints = num_keypoints)

    return model
model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh,num_keypoints = num_keypoints)
# model = detection.keypointrcnn_resnet50_fpn()
assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
weights_dict = torch.load(weights_path, map_location='cpu')
weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
model.load_state_dict(weights_dict)
# model.to(device)
model.eval()

init_img = torch.randn(3, 720, 1280)
image = cv2.imread("./test.jpg")
image = torch.tensor(image,dtype = torch.float32).permute(2,0,1).unsqueeze(0).div(255)

predictions = model(image)[0]

input_names = ["input"]
output_names = ["boxes","labels","scores","keypoints","keypoint_socores"]

# torch.onnx.export(model,image_list,"./keypoint_rcnn_r50_fpn.onnx",opset_version = 11,input_names = ["input"],output_names = ["output"])
torch.onnx.export(model,image,"./keypoint_rcnn_r50_fpn.onnx",opset_version = 11,
                    input_names = input_names,
                    output_names =output_names)


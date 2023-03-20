from openvino.runtime import Core
from openvino.offline_transformations import serialize
import cv2
import numpy as np
import time
ie = Core()
print("use core engine inference")
devices = ie.available_devices
for device in devices:
    device_name = ie.get_property(device_name = device,name = "FULL_DEVICE_NAME")
    print(f"{device}:{device_name}")


#####export onnx to bin,xml
onnx_model_path = "deploy/keypoint_rcnn_r50_fpn.onnx"  
model_onnx = ie.read_model(model = onnx_model_path)
compiled_model_onnx = ie.compile_model(model = model_onnx,device_name = "CPU")
serialize(model = model_onnx,
            model_path ="deploy/export_onnx_model.xml",
            weights_path = "deploy/export_onnx_model.bin")


model_xml = "deploy/export_onnx_model.xml"
model_bin = "deploy/export_onnx_model.bin"
model = ie.read_model(model = model_xml)
# model = ie.read_network(model = model_xml,weights = model_bin)

compiled_model = ie.compile_model(model = model,device_name = "CPU")

#input layer info
input_layer = model.input(0)
print("input layer:####\n",input_layer)
print("input layer precision:",input_layer.element_type)
print("input layer shape:##",input_layer.shape)

#output layer info
output_layer = model.output(3)
print("output layer:####\n",output_layer)
# print("output layer precision:",output_layer.element_type)
# print("output layer shape:##",output_layer.shape)


image = cv2.imread("deploy/test.jpg").transpose(2, 0, 1)[None,:].astype(np.float32)/255
time1 = time.time()
result = compiled_model([image])
print(result)
time2 = time.time()
print("openvino inference used time:##",time2-time1)
# request = compiled_model.create_infer_request()
# request.infer({input_layer.any_name:image})
# kp = request.get_tensor("keypoints").data
# print(kp)

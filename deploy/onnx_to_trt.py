import os
import tensorrt as trt
import sys

TRT_LOGGER = trt.Logger()
model_path = 'deploy/keypoint_rcnn_r50_fpn.onnx'
engine_file_path = "deploy/keypoint_rcnn_r50_fpn.trt"
# EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # batchsize=1
EXPLICIT_BATCH = []

# with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) \
#        as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network,
                                                                                                            TRT_LOGGER) as parser:
    builder.max_workspace_size = 1 << 28
    builder.max_batch_size = 1
    print(network)
    if not os.path.exists(model_path):
        print('ONNX file {} not found.'.format(model_path))
        exit(0)
    print('Loading ONNX file from path {}...'.format(model_path))
    with open(model_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print('parser.get_error(error)', parser.get_error(error))

    last_layer = network.get_layer(network.num_layers - 1)
    network.mark_output(last_layer.get_output(0))

    network.get_input(0).shape = [1, 3, 640, 160]
    print('Completed parsing of ONNX file')
    engine = builder.build_cuda_engine(network)
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
        print('save  trt success!!')

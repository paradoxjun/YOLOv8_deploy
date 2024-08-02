import onnx
import numpy as np
from onnx import helper, numpy_helper
from onnxconverter_common import convert_float_to_float16


model_input_path = './det_hand/yolov8_det_hands_s_07_11.onnx'
model_output_path = model_input_path[:-4] + 'fp16.onnx'

model = onnx.load_model(model_input_path)
model_fp16 = convert_float_to_float16(model)
onnx.save_model(model_fp16, model_output_path)

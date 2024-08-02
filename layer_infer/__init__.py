import os
import sys

# __slot__ = ['det_hand']

current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.dirname(current_directory)
if root_directory not in sys.path:
    sys.path.insert(0, root_directory)

from layer_infer.yolov8.yolov8_det import YOLOv8BaseDetector


det_hand_param = {
    "model_path": "./models/det_hand/yolov8_det_hands_s_07_11.onnx",
    "conf_thres": 0.25,
    "iou_thres": 0.7
}

det_coco_param = {
    "model_path": "./models/det_coco/yolov8m.onnx",
    "conf_thres": 0.5,
    "iou_thres": 0.7
}

det_bank_desk_param = {
    "model_path": "./models/det_bank_desk/yolov8_det_bank_desk_m_06_14.onnx",
    "conf_thres": 0.25,
    "iou_thres": 0.6
}

det_atm_broken_param = {
    "model_path": "./models/det_atm_broken/yolov8m_det_atm_broken_0731_epoch300_SGD_manual.onnx",
    "conf_thres": 0.5,
    "iou_thres": 0.6,
    "img_size": (480, 480)
}

det_hand = YOLOv8BaseDetector(**det_hand_param)
det_coco = YOLOv8BaseDetector(**det_coco_param)
det_bank_desk = YOLOv8BaseDetector(**det_bank_desk_param)
det_atm_broken = YOLOv8BaseDetector(**det_atm_broken_param)

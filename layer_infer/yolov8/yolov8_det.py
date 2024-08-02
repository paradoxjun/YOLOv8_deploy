import values.error_type as error_common
import values.result_code as result_code
from layer_infer.yolov8.utils.yolov8 import YOLOv8


_IMG_SIZE = (640, 640)
_CONF_THRES = 0.25
_IOU_THRES = 0.5


class YOLOv8BaseDetector:
    def __init__(self, model_path, img_size=_IMG_SIZE, conf_thres=_CONF_THRES, iou_thres=_IOU_THRES):
        self.det_model = YOLOv8(model_path=model_path, img_size=img_size, conf_thres=conf_thres, iou_thres=iou_thres)

    def __call__(self, image_input):
        return self.pipeline(image_input)

    def pipeline(self, image_input):
        result_state = result_code.SUCCESS.COMMON
        bbox, conf, cls, cost_time = None, None, None, []

        try:
            bbox, conf, cls, cost = self.det_model(image_input)
            cost_time = [round(x * 1000, 2) for x in cost]
            info = (f"RUN SUCCESS: Total time: {cost_time[0]} ms, Preprocess time: {cost_time[1]} ms, "
                    f"Inference time: {cost_time[2]} ms, Postprocess time: {cost_time[3]} ms. ")

        except error_common.PreProcessError as e:
            result_state = result_code.PREPROCESS_FAILED.COMMON
            info = f"RUN FAILED: exist error in preprocess: {e}. "

        except error_common.DetectionInferError as e:
            result_state = result_code.DETECT_FAILED.COMMON
            info = f"RUN FAILED: exist error in model detection: {e}. "

        except error_common.PostProcessError as e:
            result_state = result_code.POSTPROCESS_FAILED.COMMON
            info = f"RUN FAILED: exist error in postprocess: {e}. "

        except Exception as e:
            result_state = result_code.FAILED.COMMON
            info = f"RUN FAILED: {e}. "

        result = {
            'state': result_state,      # 状态码
            'bbox': [] if bbox is None else bbox.tolist(),      # 检测框：xyxy int[4]
            'conf': [] if conf is None else conf.tolist(),      # 置信度：float[1]
            'cls': [] if cls is None else cls.tolist(),         # 类别：int[1]
            'cost_time': cost_time,     # 消耗时间：float[4] ms
            'info': info,               # 描述信息：str
        }

        return result


if __name__ == '__main__':
    # MODEL_PATH = r'../../models/det_hand/yolov8_det_hands_s_07_11.onnx'
    MODEL_PATH = r'../../models/det_coco/yolov8m.onnx'
    hand_det = YOLOv8BaseDetector(MODEL_PATH, img_size=(640, 640))
    img = "https://pic4.zhimg.com/80/v2-81b33cc28e4ba869b7c2790366708e97_1440w.webp"
    res = hand_det(img)
    print(res)

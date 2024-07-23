import error_type.error_common as error_common
import values.result_code as result_code

from datetime import datetime
from values.det_class_name import hand_class_names
from yolov8_det.utils.yolov8 import YOLOv8
from yolov8_det.utils.image import draw_detections_on_raw_image, img_to_base64


_MODEL_PATH = r'../models/det_hand/hands_07_15_m.onnx'
_IMG_SIZE = (640, 640)
_CONF_THRES = 0.25
_IOU_THRES = 0.5


class HandDetector:
    def __init__(self, model_path=_MODEL_PATH, img_size=_IMG_SIZE, conf_thres=_CONF_THRES, iou_thres=_IOU_THRES):
        self.det_model = YOLOv8(model_path=model_path, img_size=img_size, conf_thres=conf_thres, iou_thres=iou_thres)

    def __call__(self, image_input, output_image=True):
        return self.pipeline(image_input, output_image)

    def pipeline(self, image_input, output_image):
        result_state = result_code.SUCCESS.COMMON
        bbox, conf, cls, cost_time = [], [], [], []
        time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        image_plot = None

        try:
            bbox, conf, cls, cost = self.det_model(image_input)
            cost_time = [round(x * 1000, 2) for x in cost]

            info = (f"RUN SUCCESS: Total time: {cost_time[0]} ms, Preprocess time: {cost_time[1]} ms, "
                    f"Inference time: {cost_time[2]} ms, Postprocess time: {cost_time[3]} ms.")

            if output_image:
                image_origin = self.det_model.get_image(image_input)
                draw_detections_on_raw_image(image_origin, bbox, conf, cls, hand_class_names)
                image_plot = img_to_base64(image_origin)

        except error_common.PreProcessError as e:
            result_state = result_code.PREPROCESS_FAILED.COMMON
            info = f"RUN FAILED: exist error in preprocess: {e}"

        except error_common.DetectionInferError as e:
            result_state = result_code.DETECT_FAILED.COMMON
            info = f"RUN FAILED: exist error in model detection: {e}"

        except error_common.PostProcessError as e:
            result_state = result_code.POSTPROCESS_FAILED.COMMON
            info = f"RUN FAILED: exist error in postprocess: {e}"

        except Exception as e:
            result_state = result_code.FAILED.COMMON
            info = f"RUN FAILED: {e}"

        result = {
            'state': result_state,      # 状态码
            'bbox': bbox.tolist(),      # 检测框：xyxy int[4]
            'conf': conf.tolist(),      # 置信度：float[1]
            'cls': cls.tolist(),        # 类别：int[1]
            'cost_time': cost_time,     # 消耗时间：float[4] ms
            'info': info,               # 描述信息：str
            'flow_no': time_string,     # 流水号：str
            'image_plot': image_plot    # 绘制的图片：str/NULL
        }

        return result


if __name__ == '__main__':
    import cv2
    from yolov8_det.utils.image import base64_to_img

    hand_det = HandDetector()
    img = "https://pic4.zhimg.com/80/v2-81b33cc28e4ba869b7c2790366708e97_1440w.webp"
    res = hand_det(img, output_image=False)
    print(res)

    if res['image_plot']:
        img_bgr = base64_to_img(res['image_plot'], rgb=False)

        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", img_bgr)
        cv2.waitKey(0)

import cv2
import onnxruntime
import time
import numpy as np
import values.error_type as error_common
from utils.compute import nms, xywh2xyxy
from layer_infer.yolov8.utils.yolov8 import YOLOv8
from utils.ops_image import get_image


class YOLOv8KEYPOINTS(YOLOv8):
    def __init__(self, model_path, img_size=(640, 640), conf_thres=0.6, iou_thres=0.7, class_name='person',
                 kpt_thres=0.5):
        super().__init__(model_path, img_size, conf_thres, iou_thres)
        self.input_height = img_size[0]
        self.input_width = img_size[1]
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.initialize_model(model_path)  # Initialize model
        self.keypoint_class_name = class_name  # 关键点所属类别名
        self.kpt_shape = [17, 3]  # 关键点形状
        self.keypoints = np.zeros(self.kpt_shape)  # 初始化单个人体关键点实例
        self.keypoint_threshold = kpt_thres  # 可视化时关键点过滤阈值, 默认threshold = 0.5

    def __call__(self, image):
        return self.person_keypoints(image)

    def initialize_model(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())
        self.get_model_details()

    def person_keypoints(self, image):
        t0 = time.perf_counter()  # start time
        try:
            input_tensor = self.prepare_input(image)

        except (error_common.ParsingUrlError, error_common.ReadImageError,
                error_common.InvalidImageError, error_common.InputFormatError) as e:
            raise error_common.PreProcessError(f"{e}: Read image failed.")

        except Exception as e:
            raise error_common.PreProcessError(f"{e}: Get input tensor failed.")

        t1 = time.perf_counter()  # preprocess time

        try:
            outputs = self.inference(input_tensor)

        except Exception as e:
            raise error_common.DetectionInferError(f"{e}: Detection model infer failed.")
        # except Exception as e:
        #     raise error_common.KeypointPredictionInferError(f"{e}: Detection model infer failed.")

        t2 = time.perf_counter()  # model infer time

        try:
            self.boxes, self.scores, self.keypoints = self.process_output(outputs)

        except Exception as e:
            raise error_common.PostProcessError(f"{e}: Post-process failed.")

        t3 = time.perf_counter()  # total time cost, and postprocess time

        # print(f"Total time: {(t3 - t0) * 1000:.2f} ms, Preprocess time: {(t1 - t0) * 1000:.2f} ms, "
        #       f"Inference time: {(t2 - t1) * 1000:.2f} ms, Postprocess time: {(t3 - t2) * 1000:.2f} ms.")

        return self.boxes, self.scores, self.keypoints, (t3 - t0, t1 - t0, t2 - t1, t3 - t2)

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.resize(image, (self.input_width, self.input_height))
        # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)    # Convert BGR to RGB

        # Scale input pixel values to 0 to 1 and transpose
        input_img = (input_img / 255.0).astype(np.float32)
        input_tensor = np.transpose(input_img, (2, 0, 1))[np.newaxis, :, :, :]

        return input_tensor

    def inference(self, input_tensor):
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T
        # [cx, cy, w, h, conf, (x, y, v) * 17] * 8400
        # print('debug -> process_output output.shape{} pred.shape{}'.format(np.shape(output), predictions.shape))
        # Filter out object confidence scores below threshold
        #

        # scores = np.max(predictions[:, 4:5], axis=1)
        # 取置信度 过滤bbox
        scores = predictions[:, 4:5].reshape((1, -1)).squeeze()
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return (np.array([], dtype=np.float32).reshape(0, 4), np.array([], dtype=np.int32),
                    np.array([], dtype=np.float32), np.array([], dtype=np.float32).reshape(*self.kpt_shape))

        # Get bounding boxes and kpt for each object

        # kpts = self.rescale_coords(kpts)
        print('debug -> process_output  pred.shape{}'.format(predictions.shape))
        boxes, kpts = self.extract_boxes_and_kpts(predictions)

        # print('debug -> process_output kpts.shape{}  boxes.shape{} kpts{} boxes{}'.format(kpts.shape, boxes.shape, kpts,
        #                                                                                 boxes))
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = nms(boxes, scores, self.iou_threshold)

        ###todo 过滤置信度小于0.5的关键点
        # keypoint (x, y, v) 网络未预测出的点依然占位为(0, 0, 0) ,
        # 所以此处做过滤意义不大, 可视化时可用v > 0.5过滤

        return boxes[indices], scores[indices], kpts[indices]

    def extract_boxes_and_kpts(self, predictions):
        boxes = predictions[:, :4]  # Extract boxes from predictions
        boxes = self.rescale_boxes(boxes)  # Scale boxes to original image dimensions
        boxes = xywh2xyxy(boxes)  # Convert boxes to xyxy format
        keypoints = self.rescale_coords(predictions[:, 5:])

        return boxes, keypoints

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])

        return boxes

    def rescale_coords(self, kpts_):
        kpts_ = np.reshape(kpts_, (-1, *self.kpt_shape))
        # print('debug -> rescale_coords kpts.shape{}\n '.format(kpts_.shape))
        input_shape = np.array([self.input_width, self.input_height, 1.0])
        image_shape = np.array([self.img_width, self.img_height, 1.0])
        kpts_ = [np.divide(pairs, input_shape, dtype=np.float32) for pairs in kpts_]
        kpts_ = [np.multiply(pairs, image_shape, dtype=np.float32) for pairs in kpts_]

        print('debug -> rescale_coords kpts.shape{}\n '.format(np.array(kpts_).shape))
        return np.reshape(kpts_, newshape=(-1, self.kpt_shape[0] * self.kpt_shape[1]))

    def get_model_details(self):
        model_inputs = self.session.get_inputs()
        model_outputs = self.session.get_outputs()

        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    from utils.ops_image import draw_person_keypoints, img_to_base64

    # 加载模型
    model_path = "../../../models/pose_person/yolov8m-pose.onnx"
    pose_predictor = YOLOv8KEYPOINTS(model_path, conf_thres=0.3, iou_thres=0.5, kpt_thres=0.5)

    # 加载图片
    img_1 = "../../../test_data/00000108.jpg"  # URL读取
    img_2 = "../../../test_data/test_hammer_02.jpg"  # base64读取
    # _, buffer = cv2.imencode(".jpg", cv2.imread(img_2))
    # img_2 = base64.b64encode(buffer).decode('utf-8')
    img_2 = img_to_base64(cv2.imread(img_2), rgb=False)
    img_3 = "../../../test_data/zidane.jpg"  # 路径读取
    img_4 = cv2.imread(img_3)  # np数组读取
    img_4 = cv2.cvtColor(img_4, cv2.COLOR_BGR2RGB)

    # 推理并绘制图片

    img_rgb = get_image(img_4)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    # cv2.imshow("Output", img_bgr)
    # cv2.waitKey(0)

    bbox, conf, kpts, _ = pose_predictor(img_rgb)

    print(
        'debug -> main bbox.shape{}\nkpts.shape{}\nconf.shape{}\n '.format(np.shape(bbox), np.shape(kpts),
                                                                           np.shape(conf)))

    img_plot = draw_person_keypoints(img_bgr, bbox, kpts, conf, 0.5, 'person')
    print('debug -> main img_plot.shape{}\n '.format(np.array(img_plot).shape))
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", img_plot)
    cv2.waitKey(0)

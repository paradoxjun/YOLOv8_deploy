import base64
import cv2
import requests
import os
import sys

import numpy as np
from PIL import Image
from io import BytesIO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

import values.error_type as error_common
from values.colors import colors_bgr as colors
from values.strings import legal_url_v1


def get_image(image_input):
    read_type = "url"

    if isinstance(image_input, str):
        if legal_url_v1.match(image_input):
            try:
                response = requests.get(image_input)
                img_rgb = np.array(Image.open(BytesIO(response.content)).convert('RGB'))

            except Exception as e:
                raise error_common.ParsingUrlError(f"{e}: Unable to parse URL.")
        else:
            try:
                img_data = base64.b64decode(image_input)
                img_rgb = np.array(Image.open(BytesIO(img_data)).convert('RGB'))
                read_type = "base64"

            except Exception:
                try:
                    img = cv2.imread(image_input)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    read_type = "image path"

                except Exception as e:
                    raise error_common.ReadImageError(f"{e}: Read image failed, get invalid input str.")

    elif isinstance(image_input, np.ndarray):
        img_rgb = image_input
    else:
        raise error_common.InputFormatError(f"Error: Get invalid input format {type(image_input)}")

    if img_rgb is None or len(img_rgb.shape) != 3:
        raise error_common.InvalidImageError(f"Error: Get invalid image (None or dim != 3), read by [{read_type}]")

    return img_rgb


def img_to_base64(image_rgb, rgb=True):
    if not rgb:
        image_rgb = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_BGR2RGB)

    image_pil = Image.fromarray(image_rgb)  # 将 RGB 张量转换为图像格式
    buffered = BytesIO()  # 将图像编码为 PNG 格式
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")  # 将编码后的图像转换为 Base64 字符串

    return img_str


def base64_to_img(base64_string, rgb=True):
    decoded_bytes = base64.b64decode(base64_string)  # 解码 Base64 字符串
    image = np.array(Image.open(BytesIO(decoded_bytes)))  # 将字节流转换为图像对象

    if not rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将 PIL 图像对象转换为 OpenCV 的 BGR 格式

    return image


def resize_image(image, new_width=None, new_height=None, keep_wh_ratio=True):
    """
    Resize an image while keeping the aspect ratio intact if required.

    :param image: Path to the input image or numpy array.
    :param new_width: Desired width of the resized image.
    :param new_height: Desired height of the resized image.
    :param keep_wh_ratio: Boolean to keep width-height ratio.
    :return: Resized image and the (width_ratio, height_ratio).
    """
    is_pil = False

    if isinstance(image, np.ndarray):
        original_height, original_width = image.shape[:2]
    else:
        original_width, original_height = image.size
        is_pil = True

    if new_width and not new_height:
        new_height = int((new_width / original_width) * original_height) if keep_wh_ratio else int(original_height)
    elif new_height and not new_width:
        new_width = int((new_height / original_height) * original_width) if keep_wh_ratio else int(original_width)
    elif new_width and new_height:
        if keep_wh_ratio:
            width_ratio = new_width / original_width
            height_ratio = new_height / original_height
            if width_ratio < height_ratio:
                new_height = int(width_ratio * original_height)
            else:
                new_width = int(height_ratio * original_width)
    elif not new_width and not new_height:
        raise ValueError("Either new_width or new_height must be provided.")

    if is_pil:
        resized_img = image.resize((new_width, new_height), Image.ANTIALIAS)
    else:
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    width_ratio = new_width / original_width
    height_ratio = new_height / original_height

    return resized_img, (width_ratio, height_ratio)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scalefill=False, scaleup=True, stride=32):
    """
    调整图片大小并填充以适应目标尺寸。
    :param im:输入图片。
    :param new_shape:目标形状，默认 (640, 640)。
    :param color:填充颜色，默认 (114, 114, 114)。
    :param auto:自动调整填充，保持最小矩形。True会让图片宽高是stride的最小整数倍，比如32，可以方便卷积。
    :param scalefill:是否拉伸填充。在auto是False时，True会让图片拉伸变形。
    :param scaleup:是否允许放大。False让图片只能缩小。
    :param stride:步幅大小，默认 32。
    :return:返回调整后的图片，缩放比例(宽，高)和填充值。
    """
    shape = im.shape[:2]  # 获取当前图片的形状 [高度, 宽度]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 缩放比例 (新尺寸 / 旧尺寸)
    if not scaleup:  # 如果不允许放大，只进行缩小 (提高验证的 mAP)
        r = min(r, 1.0)

    ratio = r, r  # 计算填充宽度和高度的缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 新的未填充尺寸 (宽度, 高度)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算宽高方向的填充值
    if auto:  # 如果设置为自动，保持最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # 使填充值是步幅的倍数
    elif scalefill:  # 如果拉伸填充，完全填充
        dw, dh = 0.0, 0.0  # 不进行填充
        new_unpad = (new_shape[1], new_shape[0])  # 未填充的尺寸就是目标尺寸
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 计算宽高的缩放比例

    dw /= 2  # 将填充值均分到两侧
    dh /= 2  # 将填充值均分到上下

    if shape[::-1] != new_unpad:  # 如果当前形状和新的未填充形状不同，则调整大小
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下填充的像素数
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右填充的像素数
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加填充边框，填充值为指定颜色

    return im, ratio, (dw, dh)  # 返回调整后的图片，缩放比例和填充值


def draw_detections_on_raw_image(image, boxes, scores, class_ids, class_names=None):
    # more fast, but lack draw mask
    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.001
    text_thickness = int(min([img_height, img_width]) * 0.002)

    for box, class_id, score in zip(boxes, class_ids, scores):
        color = colors[class_id % len(colors)]

        x1, y1, x2, y2 = list(map(int, box)) if isinstance(box, (list, tuple)) else box.astype(int)

        # Draw bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = str(int(class_id)) if class_names is None else class_names[class_id]
        caption = f'{label} {int(score * 100)}%'

        # Draw text
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=font_size, thickness=text_thickness)
        th = int(th * 1.2)
        cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

        cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    (255, 255, 255), text_thickness, cv2.LINE_AA)


def draw_detections_pipeline(image_det, boxes, scores, class_ids, class_names=None, mask_alpha=0.3):
    image = image_det.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.001
    text_thickness = int(min([img_height, img_width]) * 0.002)

    for box, class_id in zip(boxes, class_ids):
        # 必须先画，否则多次叠加胡导致不同目标颜色深浅不同
        color = colors[class_id]
        x1, y1, x2, y2 = list(map(int, box)) if isinstance(box, (list, tuple)) else box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

    image = cv2.addWeighted(image, mask_alpha, image_det, 1 - mask_alpha, 0)  # 叠加两张图

    for box, class_id, score in zip(boxes, class_ids, scores):
        color = colors[class_id]
        x1, y1, x2, y2 = list(map(int, box)) if isinstance(box, (list, tuple)) else box.astype(int)

        # Draw bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw text
        label = class_names[class_id] if class_names is not None else str(int(class_id))
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=font_size, thickness=text_thickness)
        th = int(th * 1.2)
        cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

        cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    (255, 255, 255), text_thickness, cv2.LINE_AA)

    return image


def draw_detections(image, boxes, scores, class_ids, class_names=None, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]

        draw_box(det_img, box, color)

        label = class_names[class_id] if class_names is not None else str(int(class_id))
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_person_keypoints(image, boxes, kpts, scores, visibility, label='person', mask_alpha=0.3):
    # lizhonghao 0808
    '''绘制所有人体检测框和关键点实例'''
    pose_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # print('debug -> draw_person_keypoints box.shape{} kpts.shape{}\n '.format(np.array(boxes).shape,
    #                                                                           np.array(kpts).shape))
    pose_img = draw_masks(pose_img, boxes, np.array([0] * len(boxes)), mask_alpha)
    # 从colors中取颜色
    # 关键点颜色
    kpt_color = [16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]
    # 骨骼颜色
    limb_color = [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
    # bbox颜色
    bbox_text_color = colors[0]
    for idx, (box, kpt, score) in enumerate(zip(boxes, kpts, scores)):
        draw_box(pose_img, box, bbox_text_color)
        draw_single_keypoint(pose_img, kpt, visibility, kpt_color, limb_color)

        caption = f'{label} {int(score * 100)}%'
        draw_text(pose_img, caption, box, bbox_text_color, font_size, text_thickness)
        # if idx > 5:
        #     break

    return pose_img


def draw_single_keypoint(image: np.ndarray, kpt: np.ndarray, visibility: np.float32, kpt_color: list, limb_color: list):
    from values.class_name_pose import coco_person_keypoint_class_name, coco_person_skeleton
    # lizhonghao 0808
    '''绘制单个人关键点'''
    # keypoints = box[5:]
    # single pose
    keypoints = kpt
    # print('debug -> draw_keypoint- >  keypoints{}'.format(keypoints))
    print('debug -> draw_keypoint- >  keypoints.shape{}'.format(np.shape(keypoints)))
    keypoints = np.array(keypoints).reshape(-1, 3)

    # print('debug -> draw_keypoint  kpts.shape{}\n '.format(np.array(keypoints).shape))

    for i, kp_ in enumerate(keypoints):
        # todo 此处可用索引打印出关键点名称
        x, y, conf = kp_
        # color_k = [colors[x] for x in kpt_color[i]]
        color_k = colors[kpt_color[i]]
        if conf < visibility:
            continue
        if x != 0 and y != 0:
            cv2.circle(image, (int(x), int(y)), 3, color_k, -1, lineType=cv2.LINE_AA)
            # print('debug -> draw_keypoint {}th - pair_kp_x:{}_y:{}_conf:{}\n '.format(i, x, y, conf))

    for i, sk in enumerate(coco_person_skeleton):
        pos1 = (int(keypoints[(sk[0] - 1), 0]), int(keypoints[(sk[0] - 1), 1]))
        pos2 = (int(keypoints[(sk[1] - 1), 0]), int(keypoints[(sk[1] - 1), 1]))

        conf1 = keypoints[(sk[0] - 1), 2]
        conf2 = keypoints[(sk[1] - 1), 2]
        if conf1 < visibility or conf2 < visibility:
            continue
        if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
            continue
        cv2.line(image, pos1, pos2, colors[limb_color[i]], thickness=1, lineType=cv2.LINE_AA)


def draw_box(image: np.ndarray, box: np.ndarray, color=(0, 0, 255), thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = list(map(int, box)) if isinstance(box, (list, tuple)) else box.astype(int)

    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color=(0, 0, 255), font_size: float = 0.001,
              text_thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = list(map(int, box)) if isinstance(box, (list, tuple)) else box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness,
                       cv2.LINE_AA)


def draw_masks(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = list(map(int, box)) if isinstance(box, (list, tuple)) else box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

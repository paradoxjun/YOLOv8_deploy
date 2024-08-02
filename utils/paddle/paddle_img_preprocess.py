"""
代码参考自：
    deploy/python/preprocess.py
    ppcls/data/preprocess/ops/operators.py
"""

import io
import cv2
import random
import numpy as np
from functools import partial
from PIL import Image


class OperatorParamError(ValueError):
    """ OperatorParamError
    """
    pass


def read_image(image):
    """
    自己实现读取路径或cv2的结果（BGR）。看情况再增加 PIL、base64等读取方式。
    两种读取方式不同，会导致推理结果略有差异：一个维度上的误差在 0.0001% 左右。
    """
    if isinstance(image, str):  # 从文件路径读取图片
        img_list = [cv2.imread(image)]
        if img_list[0] is None:
            raise ValueError("Error path: Could not read the image.")
    elif isinstance(image, np.ndarray):  # image 是 ndarray
        img_list = [image]
    elif isinstance(image, list):
        img_list = image
    else:
        raise ValueError("Error image format, need a image path or the result of cv2 to read.")

    return img_list


class DecodeImage(object):
    """ decode image """

    def __init__(self,
                 to_np=True,
                 to_rgb=True,
                 channel_first=False,
                 backend="cv2"):
        self.to_np = to_np  # to numpy
        self.to_rgb = to_rgb  # only enabled when to_np is True
        self.channel_first = channel_first  # only enabled when to_np is True

        if backend.lower() not in ["cv2", "pil"]:
            print(f'The backend of DecodeImage only support \"cv2\" or \"PIL\". '
                  f'\"{backend}\" is unavailable. Use \"cv2\" instead.')
            backend = "cv2"
        self.backend = backend.lower()

        if not to_np:
            print(f"\"to_rgb\" and \"channel_first\" are only enabled when to_np is True. \"to_np\" is now {to_np}.")

    def __call__(self, img):
        if isinstance(img, Image.Image):
            assert self.backend == "pil", "invalid input 'img' in DecodeImage"
        elif isinstance(img, np.ndarray):
            assert self.backend == "cv2", "invalid input 'img' in DecodeImage"
        elif isinstance(img, bytes):
            if self.backend == "pil":
                data = io.BytesIO(img)
                img = Image.open(data)
            else:
                data = np.frombuffer(img, dtype="uint8")
                img = cv2.imdecode(data, 1)
        else:
            raise ValueError("invalid input 'img' in DecodeImage")

        if self.to_np:
            if self.backend == "pil":
                assert img.mode == "RGB", f"invalid shape of image[{img.shape}]"
                img = np.asarray(img)[:, :, ::-1]  # BRG

            if self.to_rgb:
                assert img.shape[
                    2] == 3, f"invalid shape of image[{img.shape}]"
                img = img[:, :, ::-1]

            if self.channel_first:
                img = img.transpose((2, 0, 1))

        return img


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self,
                 scale=None,
                 mean=None,
                 std=None,
                 order='chw',
                 output_fp16=False,
                 channel_num=3):
        if isinstance(scale, str):
            scale = eval(scale)
        assert channel_num in [
            3, 4
        ], "channel number of input image should be set to 3 or 4."
        self.channel_num = channel_num
        self.output_dtype = 'float16' if output_fp16 else 'float32'
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if self.order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"

        img = (img.astype('float32') * self.scale - self.mean) / self.std

        if self.channel_num == 4:
            img_h = img.shape[1] if self.order == 'chw' else img.shape[0]
            img_w = img.shape[2] if self.order == 'chw' else img.shape[1]
            pad_zeros = np.zeros(
                (1, img_h, img_w)) if self.order == 'chw' else np.zeros(
                    (img_h, img_w, 1))
            img = (np.concatenate(
                (img, pad_zeros), axis=0)
                   if self.order == 'chw' else np.concatenate(
                       (img, pad_zeros), axis=2))

        return img.astype(self.output_dtype)


class UnifiedResize(object):
    def __init__(self, interpolation=None, backend="cv2", return_numpy=True):
        _cv2_interp_from_str = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'area': cv2.INTER_AREA,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
            'random': (cv2.INTER_LINEAR, cv2.INTER_CUBIC)
        }
        _pil_interp_from_str = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'box': Image.BOX,
            'lanczos': Image.LANCZOS,
            'hamming': Image.HAMMING,
            'random': (Image.BILINEAR, Image.BICUBIC)
        }

        def _cv2_resize(src, size, resample):
            if isinstance(resample, tuple):
                resample = random.choice(resample)
            return cv2.resize(src, size, interpolation=resample)

        def _pil_resize(src, size, resample, return_numpy=True):
            if isinstance(resample, tuple):
                resample = random.choice(resample)
            if isinstance(src, np.ndarray):
                pil_img = Image.fromarray(src)
            else:
                pil_img = src
            pil_img = pil_img.resize(size, resample)
            if return_numpy:
                return np.asarray(pil_img)
            return pil_img

        if backend.lower() == "cv2":
            if isinstance(interpolation, str):
                interpolation = _cv2_interp_from_str[interpolation.lower()]
            # compatible with opencv < version 4.4.0
            elif interpolation is None:
                interpolation = cv2.INTER_LINEAR
            self.resize_func = partial(_cv2_resize, resample=interpolation)
        elif backend.lower() == "pil":
            if isinstance(interpolation, str):
                interpolation = _pil_interp_from_str[interpolation.lower()]
            self.resize_func = partial(
                _pil_resize, resample=interpolation, return_numpy=return_numpy)
        else:
            print(f'The backend of Resize only support \"cv2\" or \"PIL\". \"{backend}\" '
                  f'is unavailable. Use \"cv2\" instead.')
            self.resize_func = cv2.resize

    def __call__(self, src, size):
        if isinstance(size, list):
            size = tuple(size)
        return self.resize_func(src, size)


class ResizeImage(object):
    """ resize image """

    def __init__(self,
                 size=None,
                 resize_short=None,
                 interpolation=None,
                 backend="cv2",
                 return_numpy=True):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise OperatorParamError("invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None")

        self._resize_func = UnifiedResize(
            interpolation=interpolation,
            backend=backend,
            return_numpy=return_numpy)

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            # numpy input
            img_h, img_w = img.shape[:2]
        else:
            # PIL image input
            img_w, img_h = img.size

        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        return self._resize_func(img, (w, h))

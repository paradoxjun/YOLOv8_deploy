import torch
import onnxruntime as ort
import numpy as np

from ultralytics.paddle.paddle_img_preprocess import read_image, ResizeImage, NormalizeImage
import torchvision.transforms as transforms


_transform_ops = {
    'ResizeImage': {
        'size': [224, 224],
        'resize_short': None,
        'interpolation': 'bilinear',
        'backend': 'cv2',
        'return_numpy': False
    },
    'NormalizeImage': {
        'scale': 1.0/255.0,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'order': 'hwc',
        'output_fp16': False,
        'channel_num': 3
    }
}

img_norm_torch = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_transform_ops['NormalizeImage']['mean'], _transform_ops['NormalizeImage']['std'])
        ])


img_resize = ResizeImage(size=_transform_ops['ResizeImage']['size'],
                         resize_short=_transform_ops['ResizeImage']['resize_short'],
                         interpolation=_transform_ops['ResizeImage']['interpolation'],
                         backend=_transform_ops['ResizeImage']['backend'],
                         return_numpy=_transform_ops['ResizeImage']['return_numpy'])

img_norm = NormalizeImage(scale=_transform_ops['NormalizeImage']['scale'],
                          mean=_transform_ops['NormalizeImage']['mean'],
                          std=_transform_ops['NormalizeImage']['std'],
                          order=_transform_ops['NormalizeImage']['order'],
                          output_fp16=_transform_ops['NormalizeImage']['output_fp16'],
                          channel_num=_transform_ops['NormalizeImage']['channel_num'])


def load_onnx_model(model_path, use_gpu=True):
    """
    加载ONNX模型。
    Args:
        model_path: ONNX模型文件的路径
        use_gpu: 是否使用GPU进行推理
    Returns: onnxruntime.InferenceSession: ONNX推理会话
    """
    sess_options = ort.SessionOptions()
    # sess_options.log_severity_level = 3     # 0表示最详细的日志级别
    # sess_options.inter_op_num_threads = 4   # 线程数
    # sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.enable_profiling = True

    providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']      # 选择执行提供程序
    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)       # 创建推理会话

    return session


def preprocess_image(image_path):
    """
    预处理图像以适应模型输入。输入参考：paddle的推理文件 inference_rec.yaml
    Args:
        image_path: 图像文件的路径/cv2读取结果/np数组列表
        input_shape: (tuple) 模型输入的形状 (C, H, W)
    Returns: (numpy.ndarray) 预处理后的图像
    """
    image_list = read_image(image_path)      # 使用OpenCV加载图像

    if image_list is None:
        raise FileNotFoundError(f"Can not read image.")

    # 用torch.Tensor可以快3ms
    image_batch = torch.cat([img_norm_torch(img_resize(img)).unsqueeze(0) for img in image_list], dim=0).float().numpy()

    # 变成(224, 224, 3)，归一化图像，从HWC转换到CHW格式
    # processed_images = [img_norm(img_resize(image)).transpose(2, 0, 1) for image in image_list]
    # image_batch = np.stack(processed_images, axis=0)        # 增加批量维度    (1, 3, 224, 224)

    return image_batch


def spherical_normalize(vector):
    """
    对输入向量进行球面归一化。
    """
    l2_norm = np.linalg.norm(vector)
    if l2_norm == 0:
        raise ValueError("零向量不能进行球面归一化")

    return vector / l2_norm

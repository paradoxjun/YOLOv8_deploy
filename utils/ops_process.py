def filter_detections(result, target_class_ids):
    # 将target_class_ids转换为集合以提高查找效率
    if not isinstance(target_class_ids, set):
        target_class_ids_set = set(target_class_ids)

    # 初始化过滤后的结果列表
    filtered_bboxes = []
    filtered_classes = []
    filtered_confidences = []

    # 遍历结果并过滤
    for bbox, cls, conf in zip(result['bbox'], result['cls'], result['conf']):
        if cls in target_class_ids_set:
            filtered_bboxes.append(bbox)
            filtered_classes.append(cls)
            filtered_confidences.append(conf)

    return filtered_bboxes, filtered_confidences, filtered_classes


def expand_bbox(xyxy, img_width, img_height, scale=0.1):
    # 计算宽度和高度，和中心点
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    center_x = xyxy[0] + width / 2
    center_y = xyxy[1] + height / 2

    # 增加10%的宽度和高度
    new_width = width * (1 + scale)
    new_height = height * (1 + scale)

    # 计算新的边界框坐标，并确保新的边界框坐标不超过图片的边界
    new_x1 = max(2, int(center_x - new_width / 2))
    new_y1 = max(2, int(center_y - new_height / 2))
    new_x2 = min(int(img_width) - 2, int(center_x + new_width / 2))
    new_y2 = min(int(img_height), int(center_y + new_height / 2))

    return new_x1, new_y1, new_x2, new_y2


def bbox_offset(bbox, offset_x, offset_y):
    for box in bbox:
        box[0] += offset_x
        box[1] += offset_y
        box[2] += offset_x
        box[3] += offset_y

    return bbox

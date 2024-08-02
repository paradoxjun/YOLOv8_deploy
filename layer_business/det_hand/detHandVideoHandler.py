import threading
import cv2
import queue
import json
import values.result_code as result_code
from flask import request, jsonify, Response
from datetime import datetime
from layer_infer import det_hand
from utils.ops_video import get_cap
from layer_business.det_hand.detHandHandler import det_hand_bp

# 全局变量
detecting = False
processing = False
results_queue = queue.Queue()
video_url = ""

# 条件变量
condition = threading.Condition()


# 视频流处理线程函数
def video_stream():
    global detecting, processing, results_queue, video_url
    cap = None
    while detecting:
        if not processing:
            # 暂停视频流读取，等待开始推理信号
            with condition:
                condition.wait()

        if not cap and video_url:
            cap = get_cap(video_url)

        if cap:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

            if processing:
                result = det_hand(frame)

                service_result = {
                    'state': result_code.SUCCESS.COMMON,  # 状态码
                    'bbox': result['bbox'],  # 检测框：xyxy int[4]
                    'conf': result['conf'],  # 置信度：float[1]
                    'cls': result['cls'],  # 类别：int[1]
                    'flow_no': timestamp,  # 流水号：str
                }

                results_queue.put(service_result)

    if cap:
        cap.release()
    cv2.destroyAllWindows()


# 路由：初始化并获取视频流地址
@det_hand_bp.route('/v_init', methods=['POST'])
def init_detection():
    global video_url, detecting
    data = request.json
    video_url = data.get('video_url')
    if video_url:
        detecting = True
        threading.Thread(target=video_stream).start()
        return jsonify({'state': result_code.SUCCESS.COMMON, 'info': 'INIT SUCCESS: URL received.'})
    else:
        return jsonify({'state': result_code.FAILED.SERVICE, 'info': 'INIT FAILED: URL is missing.'})


# 路由：开始检测
@det_hand_bp.route('/v_start', methods=['POST'])
def start_detection():
    global processing
    if video_url:
        with condition:
            processing = True
            condition.notify_all()
        return jsonify({'state': result_code.SUCCESS.COMMON, 'info': 'START SUCCESS: Detection started.'})
    else:
        return jsonify({'state': result_code.FAILED.SERVICE, 'info': 'START FAILED: URL not initialized.'})


# 路由：停止检测
@det_hand_bp.route('/v_stop', methods=['POST'])
def stop_detection():
    global processing
    processing = False
    return jsonify({'state': result_code.SUCCESS.COMMON, 'info': 'Detection stopped.'})


# 路由：结束视频流读取
@det_hand_bp.route('/v_end', methods=['POST'])
def end_stream():
    global detecting, processing
    processing = False
    detecting = False
    with condition:
        condition.notify_all()  # 唤醒线程以便它可以终止
    return jsonify({'state': result_code.SUCCESS.COMMON, 'info': 'Video stream ended.'})


# 路由：获取检测结果
@det_hand_bp.route('/v_results', methods=['GET'])
def get_results():
    def generate():
        while processing or not results_queue.empty():
            try:
                result = results_queue.get(timeout=1)
                yield f"data: {json.dumps(result)}\n\n"
            except queue.Empty:
                continue

    return Response(generate(), mimetype='text/event-stream')

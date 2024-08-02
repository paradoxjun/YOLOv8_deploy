import cv2
import requests
import json
from datetime import datetime

from utils.ops_video import get_cap
from utils.ops_image import resize_image

# rtsp_url = "rtsp://admin:gzjh@123@192.168.93.2:554/h264/ch1/main/av_stream"
rtsp_url = "rtsp://admin:gzjh@123@192.168.93.3:554/h264/ch1/main/av_stream"
# rtsp_url = "rtsp://admin:gzjh@123@192.168.93.4:554/h264/ch1/main/av_stream"
http_url = "http://192.168.135.161:8880/live/1/hls.m3u8"
results_url = "http://localhost:8888/det_hand/v_results"
url = rtsp_url

# 创建一个 VideoCapture 对象
cap = get_cap(url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 设置缓冲区大小
cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)  # 设置打开超时时间为 60 秒
cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 60000)  # 设置读取超时时间为 60 秒

# 初始化检测
requests.post('http://localhost:8888/det_hand/v_init', json={'video_url': url})

# 开始检测
requests.post('http://localhost:8888/det_hand/v_start')


def get_results_stream(url):
    response = requests.get(url, stream=True)
    for line in response.iter_lines():
        if line:
            data = line.decode('utf-8').replace('data: ', '')
            yield json.loads(data)


results_stream = get_results_stream(results_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 获取检测结果
    try:
        result = next(results_stream)
        if result:
            print(f"Received result: {result}")  # 调试信息
            for bbox, conf, cls in zip(result['bbox'], result['conf'], result['cls']):
                x1, y1, x2, y2 = list(map(int, bbox))
                # x1, y1, x2, y2 = bbox
                label = f'Class {cls}: {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    except StopIteration:
        print("StopIteration encountered")
        break
    except Exception as e:
        print(f"Error encountered: {e}")

    # 显示帧
    frame, _ = resize_image(frame, 900, 900)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 结束检测
requests.post('http://localhost:8888/det_hand/v_stop')

# 结束视频流读取
requests.post('http://localhost:8888/det_hand/v_end')

cap.release()
cv2.destroyAllWindows()

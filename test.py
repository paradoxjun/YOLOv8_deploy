import cv2
import requests
from utils.ops_image import get_image, img_to_base64, draw_detections_on_raw_image

b64_img = img_to_base64(cv2.imread('./test_data/910002863_1986570470.jpg'), False)

url = 'http://0.0.0.0:8888/det_hand/base'
data = {
    # 'image_input': 'https://pic4.zhimg.com/80/v2-81b33cc28e4ba869b7c2790366708e97_1440w.webp',
    'image_input': b64_img,
}

response = requests.post(url, json=data)


if __name__ == '__main__':
    if response.status_code == 200:
        result = response.json()
        print("检测结果：", result)

        img_bgr = cv2.cvtColor(get_image(data['image_input']), cv2.COLOR_RGB2BGR)
        draw_detections_on_raw_image(img_bgr, result['bbox'], result['conf'], result['cls'])

        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", img_bgr)
        cv2.waitKey(0)
    else:
        print("错误：", response.text)

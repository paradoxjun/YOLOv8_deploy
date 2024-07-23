import requests

url = 'http://192.168.141.107:8888/hand_detect/'
data = {
    'image_input': 'https://pic4.zhimg.com/80/v2-81b33cc28e4ba869b7c2790366708e97_1440w.webp',
    'output_image': True
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("检测结果：", result)
else:
    print("错误：", response.text)

if __name__ == '__main__':
    import base64
    import cv2
    import numpy as np
    from PIL import Image
    from io import BytesIO

    def base64_to_img(base64_string, rgb=True):
        decoded_bytes = base64.b64decode(base64_string)  # 解码 Base64 字符串
        image = np.array(Image.open(BytesIO(decoded_bytes)))  # 将字节流转换为图像对象

        if not rgb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将 PIL 图像对象转换为 OpenCV 的 BGR 格式

        return image

    if result.get('image_plot', None):
        img_bgr = base64_to_img(result['image_plot'], rgb=False)

        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", img_bgr)
        cv2.waitKey(0)

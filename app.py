from flask import Flask, request, jsonify, Blueprint
from yolov8_det.infer_onnx_hand import HandDetector


app = Flask(__name__)

# 创建模型实例
hand_detector = HandDetector(model_path=r'./models/det_hand/hands_07_15_m.onnx')

# 创建 Blueprint 实例
model_bp_1 = Blueprint('handDetect', __name__)


@model_bp_1.route('/', methods=['POST'])
def det_hand():
    try:
        data = request.json
        image_input = data['image_input']
        output_image = data.get('output_image', False)
        result = hand_detector(image_input, output_image)

        return jsonify(result)

    except Exception as e:
        return jsonify({'Error': str(e)}), 500


# 注册 Blueprint 到应用中，指定 URL 前缀
app.register_blueprint(model_bp_1, url_prefix='/hand_detect')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)  # 允许通过IP地址访问，调试模式开启

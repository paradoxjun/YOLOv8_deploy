from flask import Flask
from layer_business import pose_person_bp, det_hand_bp, det_coco_bp, det_bank_desk_bp, det_atm_broken_bp, det_person_bp

app = Flask(__name__)

# 注册 Blueprint 到应用中，指定 URL 前缀
app.register_blueprint(det_hand_bp, url_prefix='/det_hand')
app.register_blueprint(det_coco_bp, url_prefix='/det_coco')
app.register_blueprint(det_bank_desk_bp, url_prefix='/det_bank_desk')
app.register_blueprint(det_atm_broken_bp, url_prefix='/det_atm_broken')
app.register_blueprint(det_person_bp, url_prefix='/det_person')
app.register_blueprint(pose_person_bp, url_prefix='/pose_person')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)  # 允许通过IP地址访问，调试模式开启

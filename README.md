# YOLOv8_deploy
## 手部检测
> det_hand/base: 全图检测（只检测一次，获取所有的手）
> det_hand/by_person: 先检测出所有的人，再对每一个人检测手

## 常见类别检测（共80个类别）
> det_coco: 80个生活中常见的类别

## 银行桌面检测（验钞机、打开的款箱、关闭的款箱、钱）
> det_band_desk: 银行过验钞机模型

## 打砸ATM检测
> det_atm_broken: 打砸ATM机检测

## 人的检测
> det_person: 检测人，目前使用YOLOv8在COCO上的预训练模型
> 
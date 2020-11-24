from mmdet.apis import inference_detector, init_detector, show_result_pyplot

config="configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
ckpt='pretrainmodels/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
img='demo/qx1.jpg'


    # build the model from a config file and a checkpoint file
model = init_detector(config, ckpt, device="cuda:2")
# test a single image
result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.2)

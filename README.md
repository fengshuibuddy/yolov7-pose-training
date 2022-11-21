# yolov7-pose
Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

Pose estimation implimentation is based on [YOLO-Pose](https://arxiv.org/abs/2204.06806). 

## Dataset preparison

Download these tar file
Inside: 'labels/train2017', 'labels/val2017', train2017.txt, val2017.txt
[[Keypoints Labels of MS COCO 2017]](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-keypoints.zip)

Остальные фотки уже с data/get_coco.sh 
Там будут: 'images' , 'annotations' 

## Training

[yolov7-w6-person.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-person.pt)

``` shell
# python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --data data/coco_kpts.yaml --cfg cfg/yolov7-w6-pose.yaml --weights weights/yolov7-w6-person.pt --batch-size 128 --img 960 --kpt-label --sync-bn --device 0,1,2,3,4,5,6,7 --name yolov7-w6-pose --hyp data/hyp.pose.yaml
python train.py --kpt-label --sync-bn --img 640
```

1. В Датасете labels могут тупить – я скачал с репо эти фото, а фотки обычный с coco
2. Обязательно давай --kpt-lable, и --sync-bn 
3. Входной image file – обязательно должен быть 640x640 
4. Удаляй cached train, val files
5. В "loss.py" → 187, 189 line → gain = torch.ones(41, device=targets.device).long()  → добавь long
6. “./utils/plot.py” → 84 line → там “plot_skeleton_kpts” функцию положи в try-except case

## Debugging
Видать там keypoint NaN какой-нить, поэтому он не может plot его в image. Это проблема, когда ранится на Тестовых данных – Я понял что случилось:
У меня batch_size=1, значит он апдейтит каждый раз 
Когда он тренит на 2-х фотках – он еще не успевает нормально фотки потренить, поэтому веса еще не тупят (на каждой фотке делает апдейт – batch_size=1) 
Когда он тренит на больших фотках (от 10 и выше) –  он успевает нормально веса так потвикать, что в итоге Pose Extractor становится не такой идеальный.
От этого когда ранится на Тестовых данных, где-то "Keypoints - NaN" и выходят ошибки ("cannot convert float infinity to integer", "RuntimeWarning: invalid value encountered in double_scalars" еще такой есть)   


## Deploy
TensorRT:[https://github.com/nanmi/yolov7-pose](https://github.com/nanmi/yolov7-pose)

## Testing

[yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

``` shell
python test.py --data data/coco_kpts.yaml --img 960 --conf 0.001 --iou 0.65 --weights yolov7-w6-pose.pt --kpt-label
```

## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

## Acknowledgements

https://github.com/WongKinYiu/yolov7/tree/pose

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>

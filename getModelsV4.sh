mkdir -p ./data/yolov4

wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -O ./data/yolov4/yolov4.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg -O ./data/yolov4/yolov4.cfg
wget https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O ./data/yolov3/coco.names

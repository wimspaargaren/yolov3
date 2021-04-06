mkdir -p ./data/yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O ./data/yolov3/yolov3.weights
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true -O ./data/yolov3/yolov3.cfg
wget https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O ./data/yolov3/coco.names

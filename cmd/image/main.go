package main

import (
	"flag"
	"os"
	"path"

	log "github.com/sirupsen/logrus"
	"gocv.io/x/gocv"

	"github.com/wimspaargaren/yolov3"
)

var (
	yolov3WeightsPath = path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov3/data/yolov3/yolov3.weights")
	yolov3ConfigPath  = path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov3/data/yolov3/yolov3.cfg")
	cocoNamesPath     = path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov3/data/yolov3/coco.names")
)

func main() {
	imagePath := flag.String("i", path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov3/data/example_images/bird.jpg"), "specify the image path")
	flag.Parse()

	yolonet, err := yolov3.NewNet(yolov3WeightsPath, yolov3ConfigPath, cocoNamesPath)
	if err != nil {
		log.WithError(err).Fatal("unable to create yolo net")
	}

	// Gracefully close the net when the program is done
	defer func() {
		err := yolonet.Close()
		if err != nil {
			log.WithError(err).Error("unable to gracefully close yolo net")
		}
	}()

	frame := gocv.IMRead(*imagePath, gocv.IMReadColor)

	detections, err := yolonet.GetDetections(frame)
	if err != nil {
		log.WithError(err).Fatal("unable to retrieve predictions")
	}

	yolov3.DrawDetections(&frame, detections)

	window := gocv.NewWindow("Result Window")
	defer func() {
		err := window.Close()
		if err != nil {
			log.WithError(err).Error("unable to close window")
		}
	}()

	window.IMShow(frame)
	window.ResizeWindow(872, 585)

	window.WaitKey(10000000000)
}

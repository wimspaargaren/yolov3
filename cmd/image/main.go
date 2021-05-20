package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"

	log "github.com/sirupsen/logrus"
	"github.com/wimspaargaren/yolov3"
	"gocv.io/x/gocv"
)

const (
	yolov3WeightsPath = "../../data/yolov3/yolov3.weights"
	yolov3ConfigPath  = "../../data/yolov3/yolov3.cfg"
	cocoNamesPath     = "../../data/yolov3/coco.names"
)

func main() {
	imagePath := flag.String("i", "../../data/example_images/bird.jpg", "specify the image path")
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

	drawDetections(&frame, detections)

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

func drawDetections(frame *gocv.Mat, detections []yolov3.ObjectDetection) {
	for i := 0; i < len(detections); i++ {
		detection := detections[i]
		text := fmt.Sprintf("%s:%.2f", detection.ClassName, detection.Confidence)

		// Create bounding box of object
		blue := color.RGBA{0, 0, 255, 0}
		gocv.Rectangle(frame, detection.BoundingBox, blue, 3)

		// Add text background
		black := color.RGBA{0, 0, 0, 0}
		size := gocv.GetTextSize(text, gocv.FontHersheySimplex, 0.5, 1)
		r := detection.BoundingBox
		textBackground := image.Rect(r.Min.X, r.Min.Y-size.Y-1, r.Min.X+size.X, r.Min.Y)
		gocv.Rectangle(frame, textBackground, black, int(gocv.Filled))

		// Add text
		pt := image.Pt(r.Min.X, r.Min.Y-4)
		white := color.RGBA{255, 255, 255, 0}
		gocv.PutText(frame, text, pt, gocv.FontHersheySimplex, 0.5, white, 1)
	}
}

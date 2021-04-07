package main

import (
	"fmt"
	"image"
	"image/color"

	log "github.com/sirupsen/logrus"
	"github.com/wimspaargaren/yolov3"
	"gocv.io/x/gocv"
)

const (
	vehicleWeights = "../../data/yolov3/yolov3.weights"
	vehicleNetcfg  = "../../data/yolov3/yolov3.cfg"
	cocoNames      = "../../data/yolov3/coco.names"
)

func main() {
	yolonet, err := yolov3.NewNet(vehicleWeights, vehicleNetcfg, cocoNames)
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

	videoCapture, err := gocv.OpenVideoCapture(0)
	if err != nil {
		log.WithError(err).Fatal("unable to start video capture")
	}

	window := gocv.NewWindow("Result Window")
	defer func() {
		err := window.Close()
		if err != nil {
			log.WithError(err).Error("unable to close window")
		}
	}()

	frame := gocv.NewMat()
	defer func() {
		err := frame.Close()
		if err != nil {
			log.WithError(err).Errorf("unable to close image")
		}
	}()

	for {
		if ok := videoCapture.Read(&frame); !ok {
			log.Error("unable to read videostram")
		}
		if frame.Empty() {
			continue
		}
		detections, err := yolonet.GetDetections(frame)
		if err != nil {
			log.WithError(err).Fatal("unable to retrieve predictions")
		}

		drawDetections(&frame, detections)

		window.IMShow(frame)
		window.WaitKey(1)
	}
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

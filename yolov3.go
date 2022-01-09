// Package yolov3 provides a Go implementation of the YOLO V3 object detection system: https://pjreddie.com/darknet/yolo/.
//
// The yolov3 package leverages gocv(https://github.com/hybridgroup/gocv) for a neural net able to detect object.
//
// In order for the neural net to be able to detect objects, it needs the pre-trained network model
// consisting of a .cfg file and a .weights file. Using the Makefile provied by the library, these models
// can simply be downloaded by running 'make models'.
//
// In order to use the package, make sure you've checked the prerequisites in the README: https://github.com/wimspaargaren/yolov3#prerequisites
package yolov3

import (
	"fmt"
	"image"
	"image/color"
	"os"
	"strings"

	"gocv.io/x/gocv"

	"github.com/wimspaargaren/yolov3/internal/ml"
)

// Default constants for initialising the yolov3 net.
const (
	DefaultInputWidth  = 416
	DefaultInputHeight = 416

	DefaultConfThreshold float32 = 0.5
	DefaultNMSThreshold  float32 = 0.4
)

// Config can be used to customise the settings of the neural network used for object detection.
type Config struct {
	// InputWidth & InputHeight are used to determine the input size of the image for the network
	InputWidth  int
	InputHeight int
	// ConfidenceThreshold can be used to determine the minimum confidence before an object is considered to be "detected"
	ConfidenceThreshold float32
	// Non-maximum suppression threshold used for removing overlapping bounding boxes
	NMSThreshold float32

	// Type on which the network will be executed
	NetTargetType  gocv.NetTargetType
	NetBackendType gocv.NetBackendType

	// NewNet function can be used to inject a custom neural net
	NewNet func(weightsPath, configPath string) ml.NeuralNet
}

// validate ensures that the basic fields of the config are set
func (c *Config) validate() {
	if c.NewNet == nil {
		c.NewNet = initializeNet
	}
	if c.InputWidth == 0 {
		c.InputWidth = DefaultInputWidth
	}
	if c.InputHeight == 0 {
		c.InputHeight = DefaultInputHeight
	}
}

// DefaultConfig used to create a working yolov3 net out of the box.
func DefaultConfig() Config {
	return Config{
		InputWidth:          DefaultInputWidth,
		InputHeight:         DefaultInputHeight,
		ConfidenceThreshold: DefaultConfThreshold,
		NMSThreshold:        DefaultNMSThreshold,
		NetTargetType:       gocv.NetTargetCPU,
		NetBackendType:      gocv.NetBackendDefault,
		NewNet:              initializeNet,
	}
}

// ObjectDetection represents information of an object detected by the neural net.
type ObjectDetection struct {
	ClassID     int
	ClassName   string
	BoundingBox image.Rectangle
	Confidence  float32
}

// Net the yolov3 net.
type Net interface {
	Close() error
	GetDetections(gocv.Mat) ([]ObjectDetection, error)
	GetDetectionsWithFilter(gocv.Mat, map[string]bool) ([]ObjectDetection, error)
}

// yoloNet the net implementation.
type yoloNet struct {
	net       ml.NeuralNet
	cocoNames []string

	DefaultInputWidth   int
	DefaultInputHeight  int
	confidenceThreshold float32
	DefaultNMSThreshold float32
}

// NewNet creates new yolo net for given weight path, config and coconames list.
func NewNet(weightsPath, configPath, cocoNamePath string) (Net, error) {
	return NewNetWithConfig(weightsPath, configPath, cocoNamePath, DefaultConfig())
}

// NewNetWithConfig creates new yolo net with given config.
func NewNetWithConfig(weightsPath, configPath, cocoNamePath string, config Config) (Net, error) {
	if _, err := os.Stat(weightsPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("path to net weights not found")
	}

	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("path to net config not found")
	}

	cocoNames, err := getCocoNames(cocoNamePath)
	if err != nil {
		return nil, err
	}

	config.validate()

	net := config.NewNet(weightsPath, configPath)

	err = setNetTargetTypes(net, config)
	if err != nil {
		return nil, err
	}

	return &yoloNet{
		net:                 net,
		cocoNames:           cocoNames,
		DefaultInputWidth:   config.InputWidth,
		DefaultInputHeight:  config.InputHeight,
		confidenceThreshold: config.ConfidenceThreshold,
		DefaultNMSThreshold: config.NMSThreshold,
	}, nil
}

// initializeNet default method for creating neural network, leveraging gocv.
func initializeNet(weightsPath, configPath string) ml.NeuralNet {
	net := gocv.ReadNet(weightsPath, configPath)
	return &net
}

func setNetTargetTypes(net ml.NeuralNet, config Config) error {
	err := net.SetPreferableBackend(config.NetBackendType)
	if err != nil {
		return err
	}

	err = net.SetPreferableTarget(config.NetTargetType)
	if err != nil {
		return err
	}
	return nil
}

// Close closes the net.
func (y *yoloNet) Close() error {
	return y.net.Close()
}

// GetDetections retrieve predicted detections from given matrix.
func (y *yoloNet) GetDetections(frame gocv.Mat) ([]ObjectDetection, error) {
	return y.GetDetectionsWithFilter(frame, make(map[string]bool))
}

// GetDetectionsWithFilter allows you to detect objects, but filter out a given list of coco name ids.
func (y *yoloNet) GetDetectionsWithFilter(frame gocv.Mat, classIDsFilter map[string]bool) ([]ObjectDetection, error) {
	fl := []string{"yolo_82", "yolo_94", "yolo_106"}
	blob := gocv.BlobFromImage(frame, 1.0/255.0, image.Pt(y.DefaultInputWidth, y.DefaultInputHeight), gocv.NewScalar(0, 0, 0, 0), true, false)
	// nolint: errcheck
	defer blob.Close()
	y.net.SetInput(blob, "data")

	outputs := y.net.ForwardLayers(fl)

	detections, err := y.processOutputs(frame, outputs, classIDsFilter)
	if err != nil {
		return nil, err
	}

	for i, _ := range outputs {
		_ = outputs[i].Close()
	}

	return detections, nil
}

// processOutputs process detected rows in the outputs.
func (y *yoloNet) processOutputs(frame gocv.Mat, outputs []gocv.Mat, filter map[string]bool) ([]ObjectDetection, error) {
	detections := []ObjectDetection{}
	bboxes := []image.Rectangle{}
	confidences := []float32{}
	for i := 0; i < len(outputs); i++ {
		data, err := outputs[i].DataPtrFloat32()
		if err != nil {
			return nil, err
		}
		for x := 0; x < outputs[i].Total(); x += outputs[i].Cols() {
			row := data[x : x+outputs[i].Cols()]
			scores := row[5:]
			classID, confidence := getClassIDAndConfidence(scores)
			if y.isFiltered(classID, filter) {
				continue
			}
			if confidence > y.confidenceThreshold {
				confidences = append(confidences, confidence)

				boundingBox := calculateBoundingBox(frame, row)
				bboxes = append(bboxes, boundingBox)
				detections = append(detections, ObjectDetection{
					ClassID:     classID,
					ClassName:   y.cocoNames[classID],
					BoundingBox: boundingBox,
					Confidence:  confidence,
				})
			}
		}
	}
	if len(bboxes) == 0 {
		return detections, nil
	}
	indices := make([]int, len(bboxes))

	gocv.NMSBoxes(bboxes, confidences, y.confidenceThreshold, y.DefaultNMSThreshold, indices)
	result := []ObjectDetection{}
	for i, indice := range indices {
		// If we encounter value 0 skip the detection
		// except for the first indice
		if i != 0 && indice == 0 {
			continue
		}
		result = append(result, detections[indice])
	}
	return result, nil
}

func (y *yoloNet) isFiltered(classID int, classIDs map[string]bool) bool {
	if classIDs == nil {
		return false
	}
	return classIDs[y.cocoNames[classID]]
}

// calculateBoundingBox calculate the bounding box of the detected object.
func calculateBoundingBox(frame gocv.Mat, row []float32) image.Rectangle {
	if len(row) < 4 {
		return image.Rect(0, 0, 0, 0)
	}
	centerX := int(row[0] * float32(frame.Cols()))
	centerY := int(row[1] * float32(frame.Rows()))
	width := int(row[2] * float32(frame.Cols()))
	height := int(row[3] * float32(frame.Rows()))
	left := (centerX - width/2)
	top := (centerY - height/2)
	return image.Rect(left, top, left+width, top+height)
}

// getClassID retrieve class id from given row.
func getClassIDAndConfidence(x []float32) (int, float32) {
	res := 0
	max := float32(0.0)
	for i, y := range x {
		if y > max {
			max = y
			res = i
		}
	}
	return res, max
}

// getCocoNames read coconames from given path.
func getCocoNames(path string) ([]string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return strings.Split(string(content), "\n"), nil
}

// DrawDetections draws a given list of object detections on a gocv Matrix.
func DrawDetections(frame *gocv.Mat, detections []ObjectDetection) {
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

// Package ml is used as interface on how a neural network should behave
package ml

import "gocv.io/x/gocv"

// NeuralNet is the interface representing the neural network
// used for calculating the object detections
type NeuralNet interface {
	SetPreferableBackend(backend gocv.NetBackendType) error
	SetPreferableTarget(target gocv.NetTargetType) error
	SetInput(blob gocv.Mat, name string)
	ForwardLayers(outBlobNames []string) (blobs []gocv.Mat)
	Close() error
}

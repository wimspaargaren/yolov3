package yolov3

import (
	"fmt"
	"testing"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/suite"
	"github.com/wimspaargaren/yolov3/internal/ml"
	"github.com/wimspaargaren/yolov3/internal/ml/mocks"
)

type YoloTestSuite struct {
	suite.Suite

	neuralNetMock *mocks.MockNeuralNet
}

func TestYoloTestSuite(t *testing.T) {
	suite.Run(t, new(YoloTestSuite))
}

func (s *YoloTestSuite) SetupTest() {
	controller := gomock.NewController(s.T())
	s.neuralNetMock = mocks.NewMockNeuralNet(controller)
}

func (s *YoloTestSuite) TearDownSuite() {
}

func (s *YoloTestSuite) TestCorrectImplementation() {
	var _ Net = &yoloNet{}
}

func (s *YoloTestSuite) TestNewDefaultNetCorrectCreation() {
	net, err := NewNet("data/yolov3/yolov3.weights", "data/yolov3/yolov3.cfg", "data/yolov3/coco.names")
	s.Require().NoError(err)
	yoloNet := net.(*yoloNet)

	s.NotNil(yoloNet.net)
	s.Equal(81, len(yoloNet.cocoNames))
	s.Equal(inputWidth, yoloNet.inputWidth)
	s.Equal(inputHeight, yoloNet.inputHeight)
	s.Equal(confThreshold, yoloNet.confidenceThreshold)
	s.Equal(nmsThreshold, yoloNet.nmsThreshold)

	s.NoError(yoloNet.Close())
}

func (s *YoloTestSuite) TestUnableTocCreateNewNet() {
	tests := []struct {
		Name               string
		WeightsPath        string
		ConfigPath         string
		CocoNamePath       string
		Config             Config
		Error              error
		SetupNeuralNetMock func() *mocks.MockNeuralNet
	}{
		{
			Name:         "Non existent weights path",
			WeightsPath:  "data/yolov3/notexistent",
			ConfigPath:   "data/yolov3/yolov3.cfg",
			CocoNamePath: "data/yolov3/coco.names",
			Error:        fmt.Errorf("path to net weights not found"),
		},
		{
			Name:         "Non existent config path",
			WeightsPath:  "data/yolov3/yolov3.weights",
			ConfigPath:   "data/yolov3/notexistent",
			CocoNamePath: "data/yolov3/coco.names",
			Error:        fmt.Errorf("path to net config not found"),
		},
		{
			Name:         "Non existent coco names path",
			WeightsPath:  "data/yolov3/yolov3.weights",
			ConfigPath:   "data/yolov3/yolov3.cfg",
			CocoNamePath: "data/yolov3/notexistent",
		},
		{
			Name:         "Unable to set preferable backend",
			WeightsPath:  "data/yolov3/yolov3.weights",
			ConfigPath:   "data/yolov3/yolov3.cfg",
			CocoNamePath: "data/yolov3/coco.names",
			SetupNeuralNetMock: func() *mocks.MockNeuralNet {
				controller := gomock.NewController(s.T())
				neuralNetMock := mocks.NewMockNeuralNet(controller)
				neuralNetMock.EXPECT().SetPreferableBackend(gomock.Any()).Return(fmt.Errorf("very broken")).Times(1)
				return neuralNetMock
			},
			Error: fmt.Errorf("very broken"),
		},
		{
			Name:         "Unable to set preferable target type",
			WeightsPath:  "data/yolov3/yolov3.weights",
			ConfigPath:   "data/yolov3/yolov3.cfg",
			CocoNamePath: "data/yolov3/coco.names",
			SetupNeuralNetMock: func() *mocks.MockNeuralNet {
				controller := gomock.NewController(s.T())
				neuralNetMock := mocks.NewMockNeuralNet(controller)
				neuralNetMock.EXPECT().SetPreferableBackend(gomock.Any()).Return(nil).Times(1)
				neuralNetMock.EXPECT().SetPreferableTarget(gomock.Any()).Return(fmt.Errorf("very broken")).Times(1)
				return neuralNetMock
			},
			Error: fmt.Errorf("very broken"),
		},
	}

	for _, test := range tests {
		s.Run(test.Name, func() {
			test.Config.newNet = func(string, string) ml.NeuralNet {
				return test.SetupNeuralNetMock()
			}
			_, err := NewNetWithConfig(test.WeightsPath, test.ConfigPath, test.CocoNamePath, test.Config)
			s.Error(err)
			if test.Error != nil {
				s.Equal(test.Error, err)
			}
		})
	}
}

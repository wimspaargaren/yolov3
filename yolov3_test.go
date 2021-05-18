package yolov3

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/suite"
)

type YoloTestSuite struct {
	suite.Suite
}

func TestYoloTestSuite(t *testing.T) {
	suite.Run(t, new(YoloTestSuite))
}

func (s *YoloTestSuite) SetupSuite() {
}

func (s *YoloTestSuite) TearDownSuite() {
}

func (s *YoloTestSuite) TestWithContext() {
	fmt.Println("TODO")
}

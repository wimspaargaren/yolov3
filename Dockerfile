FROM gocv/opencv:4.10.0

# Install dependencies
RUN curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin v1.53.2

COPY . /go/src/github.com/wimspaargaren/yolov5

WORKDIR /go/src/github.com/wimspaargaren/yolov5
# In order to test a happy flow we need to have the actual config and weights
RUN make models

FROM gocv/opencv:4.5.2

# Install dependencies
RUN curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin v1.40.1
RUN go get -u github.com/rakyll/gotest

COPY . /go/src/github.com/wimspaargaren/yolov3

WORKDIR /go/src/github.com/wimspaargaren/yolov3

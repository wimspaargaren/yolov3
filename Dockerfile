FROM gocv/opencv:4.5.2

# Install dependencies
RUN go get -u github.com/rakyll/gotest

COPY . /go/src/github.com/wimspaargaren/yolov3

WORKDIR /go/src/github.com/wimspaargaren/yolov3

# Go YOLO V3

This repository provides an implementation of the [Yolo V3](https://pjreddie.com/darknet/yolo/) object detection system in Go leveraging [gocv](https://github.com/hybridgroup/gocv).

# Before you begin

Since this implementation builds on top of the [gocv](https://github.com/hybridgroup/gocv) library, make sure you either use one of the provided [docker images](https://github.com/hybridgroup/gocv/blob/release/Dockerfile) to run the example, or install the opencv dependencies on your system.

Furthermore, make sure you've got the yolov3 models downloaded before running the examples. Simply run `$ make models`

# Run the example

Execute the bird example:

`$ make bird-example`

Output

<img src="data/example_outputs/birds-output.png"
     alt="birds output"/>

Execute the street example:

`$ make street-example`

Output

<img src="data/example_outputs/street-output.png"
     alt="street output"/>

# CUDA

If you're interested in running yolo in Go with CUDA support, check the `cmd/example_cuda` to see a dummy example and test results of running object detection at 50 fps.

# Issues

If you have any issues, feel free to open a PR or create an issue!
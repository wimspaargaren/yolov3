.PHONY: all

data/yolov3:
	@$(shell ./getModels.sh)

# Retrieves yolov3 models
models: | data/yolov3
	@

# Runs lint
lint:
	@echo Linting...
	@golangci-lint  -v --concurrency=3 --config=.golangci.yml --issues-exit-code=0 run \
	--out-format=colored-line-number

# Runs the examples
bird-example:
	@cd cmd/image && go run .
	
street-example:
	@cd cmd/image && go run . -i ../../data/example_images/street.jpg

webcam-example:
	@cd cmd/webcam && go run .

cuda-example:
	@cd cmd/cuda && go run .

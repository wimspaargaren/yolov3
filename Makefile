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

# Run tests
test:
	@echo Running tests...
	@mkdir -p reports
	LOGFORMAT=ASCII gotest -covermode=count -p=4 -v -coverprofile reports/codecoverage_all.cov --tags=${GO_TEST_BUILD_TAGS} `go list ./...`
	@echo "Done running tests"
	@go tool cover -func=reports/codecoverage_all.cov > reports/functioncoverage.out
	@go tool cover -html=reports/codecoverage_all.cov -o reports/coverage.html
	@echo "View report at $(PWD)/reports/coverage.html"
	@tail -n 1 reports/functioncoverage.out 

# Runs the examples
bird-example:
	@cd cmd/image && go run .
	
street-example:
	@cd cmd/image && go run . -i ../../data/example_images/street.jpg

webcam-example:
	@cd cmd/webcam && go run .

cuda-example:
	@cd cmd/cuda && go run .

# CI commands
ci-init:
	@docker build -t yolov3-ci .

ci-test:
	@docker run yolov3-ci make test

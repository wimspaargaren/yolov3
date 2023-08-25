.PHONY: all test lint bird-example street-example cuda-example ci-init ci-lint ci-test

data/yolov3:
	@$(shell ./getModels.sh)

# Retrieves yolov3 models
models: | data/yolov3
	@

# Runs lint
lint:
	@echo Linting...
	@golangci-lint  -v --concurrency=3 --config=.golangci.yml --issues-exit-code=1 run \
	--out-format=colored-line-number 

# Run tests
test:
	@mkdir -p reports
	@go test -coverprofile=reports/codecoverage_all.cov ./... -cover -race -p=4
	@go tool cover -func=reports/codecoverage_all.cov > reports/functioncoverage.out
	@go tool cover -html=reports/codecoverage_all.cov -o reports/coverage.html
	@echo "View report at $(PWD)/reports/coverage.html"
	@tail -n 1 reports/functioncoverage.out 

# Opens created coverage report in default browser
coverage-report:
	@open reports/coverage.html

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

ci-lint:
	@docker run yolov3-ci make lint

ci-test:
	@docker run yolov3-ci make test

$(GOBIN)/gofumpt:
	@GO111MODULE=on go get mvdan.cc/gofumpt
	@go mod tidy

gofumpt: | $(GOBIN)/gofumpt
	@gofumpt -w

gci:
	@gci -local="github.com/wimspaargaren/yolov3" -w $(shell ls  -d $(PWD)/* | grep -v mocks)
	@gci -local="github.com/wimspaargaren/yolov3" -w $(shell ls  -d $(PWD)/cmd/*)

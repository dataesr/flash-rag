CURRENT_VERSION=$(shell grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
DOCKER_IMAGE_NAME=dataesr/rag
GHCR_IMAGE_NAME=ghcr.io/$(DOCKER_IMAGE_NAME)

build:
	@echo "Building Docker image $(GHCR_IMAGE_NAME):$(CURRENT_VERSION)"
	docker build -t $(GHCR_IMAGE_NAME):$(CURRENT_VERSION) -t $(GHCR_IMAGE_NAME):latest .
	@echo "Docker image $(GHCR_IMAGE_NAME):$(CURRENT_VERSION) built successfully"

push:
	@echo "Pushing Docker image $(GHCR_IMAGE_NAME):$(CURRENT_VERSION)"
	docker push $(GHCR_IMAGE_NAME):$(CURRENT_VERSION)
	docker push $(GHCR_IMAGE_NAME):latest
	@echo "Docker image $(GHCR_IMAGE_NAME):$(CURRENT_VERSION) pushed successfully"

run:
	@echo "Running Docker image $(GHCR_IMAGE_NAME):$(CURRENT_VERSION)"
	docker run -p 8000:8000 -d $(GHCR_IMAGE_NAME):$(CURRENT_VERSION)

stop:
	@echo "Stopping Docker image $(GHCR_IMAGE_NAME):$(CURRENT_VERSION)"
	docker stop $(GHCR_IMAGE_NAME):$(CURRENT_VERSION)

build-push:
	@"$(MAKE)" build
	@"$(MAKE)" push

release:
ifndef VERSION
	$(error VERSION is not defined. Use 'make release VERSION=x.y.z')
endif
	sed -i 's/^version = ".*"/version = "${VERSION}"/' pyproject.toml
	git commit -am '[release] version ${VERSION}'
	git tag v${VERSION}
	@echo "If everything is OK, you can push with tags i.e. git push origin main --tags"
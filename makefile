UNAME_S := $(shell uname -s)

.PHONY: preprocess benchmark compare deploy clean test

preprocess:
ifeq ($(UNAME_S),Darwin)
	docker compose --profile cpu up preprocess --build
else
	docker compose --profile gpu up preprocess --build
endif

benchmark:
ifeq ($(UNAME_S),Darwin)
	docker compose --profile cpu up cpu --build
else
	docker compose --profile gpu up pytorch cuda --build
endif

compare:
	docker compose --profile cpu up compare --build

deploy:
	bash gcp/deploy.sh

test:
	docker compose --profile cpu run --rm cpu python -m pytest quality/ -v

clean:
	docker compose --profile cpu down --rmi all 2>/dev/null || true
	docker compose --profile gpu down --rmi all 2>/dev/null || true

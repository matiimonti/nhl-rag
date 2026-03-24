.PHONY: up down build restart ps logs logs-fastapi logs-ingestion shell-fastapi shell-ingestion test

up:
	docker compose --profile ui up --build -d

down:
	docker compose --profile ui down

build:
	docker compose --profile ui build

restart:
	docker compose --profile ui restart

ps:
	docker compose ps

logs:
	docker compose --profile ui logs -f

logs-fastapi:
	docker compose logs -f fastapi

logs-ingestion:
	docker compose logs -f ingestion

shell-fastapi:
	docker compose exec fastapi bash

shell-ingestion:
	docker compose exec ingestion bash

shell-redis:
	docker compose exec redis sh

test:
	pytest tests/test_networking.py -v
	python tests/test_container_networking.py

.PHONY: up down logs build ps shell test-networking test-redis-streams

up:
	docker compose --profile ui up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose --profile ui build

ps:
	docker compose ps

shell:
	docker compose exec $(SERVICE) bash

test-networking:
	bash scripts/test_networking.sh

test-redis-streams:
	bash scripts/test_redis_streams.sh



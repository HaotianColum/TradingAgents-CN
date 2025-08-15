docker compose build web
docker compose up -d --force-recreate --no-deps web
docker compose ps web
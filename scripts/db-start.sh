#!/usr/bin/env bash
set -e

docker run -d \
  --name arxiv-pg \
  -e POSTGRES_USER=arxiv \
  -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=arxiv_agent \
  -p 5432:5432 \
  pgvector/pgvector:pg16

echo "Waiting for PostgreSQL to be ready..."
until docker exec arxiv-pg pg_isready -U arxiv -q; do
  sleep 1
done
echo "PostgreSQL ready."

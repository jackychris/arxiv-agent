#!/usr/bin/env bash
set -e

docker stop arxiv-pg
docker rm arxiv-pg
echo "PostgreSQL stopped and removed."

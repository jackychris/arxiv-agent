#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/runs/services"
PID_DIR="$ROOT/runs/services"
PYTHON_BIN="${RESEARCH_AGENT_PYTHON:-$HOME/miniconda3/envs/research-agent/bin/python}"
mkdir -p "$LOG_DIR" "$PID_DIR"

start_service() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${name}.log"
  local pid_file="$PID_DIR/${name}.pid"
  local cmd
  local quoted_root

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name already running (pid $(cat "$pid_file"))"
    return
  fi

  printf -v cmd '%q ' "$@"
  printf -v quoted_root '%q' "$ROOT"

  echo "Starting $name..."
  nohup /bin/bash -lc "cd $quoted_root && exec $cmd" >"$log_file" 2>&1 &
  echo $! >"$pid_file"
  echo "$name started (pid $!)"
}

echo "Starting professional MCP services with Docker Compose..."
docker compose up -d academic-mcp github-mcp web-mcp
start_service "research-agent" "$PYTHON_BIN" -m researcher.server
start_service "main" "$PYTHON_BIN" -m main

echo "Logs: $LOG_DIR"

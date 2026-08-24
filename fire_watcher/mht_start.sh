#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${MHT_SOURCE:-/dev/video0}"
SOCK="${MHT_SOCK:-$ROOT/fire_watcher/watcher.sock}"
LOG="${MHT_LOG:-$ROOT/fire_watcher/mht.log}"
PID="${MHT_PID:-$ROOT/fire_watcher/mht.pid}"
nohup "$ROOT/.venv/bin/python3" "$ROOT/fire_watcher/mht_live.py" --sock "$SOCK" --source "$SOURCE" >> "$LOG" 2>&1 &
echo $! > "$PID"

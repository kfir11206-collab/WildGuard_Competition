#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID="${MHT_PID:-$ROOT/fire_watcher/mht.pid}"
[ -f "$PID" ] || exit 0
P=$(cat "$PID")
kill -TERM "$P" 2>/dev/null
while kill -0 "$P" 2>/dev/null; do sleep 0.2; done
rm -f "$PID"

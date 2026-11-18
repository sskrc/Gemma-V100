#!/bin/bash
# Gemma 4 (12B + E4B, MTP) router-mode server.  Usage: ./serve.sh {start|stop|restart|status}
D="$(cd "$(dirname "$0")" && pwd)"
BIN="$D/llama.cpp/build/bin/llama-server"
export LD_LIBRARY_PATH="$D/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PORT=8100
HOST=0.0.0.0        # ローカル専用にするなら 127.0.0.1
LOG="$D/router.log"
KEYFILE="${LLAMA_API_KEY_FILE:-$D/.api_key}"   # llama.cpp 標準の --api-key-file 用。無ければ自動生成 (環境変数 LLAMA_API_KEY があればそれで上書き)
[ -n "${LLAMA_API_KEY:-}" ] && (umask 077; printf "%s\n" "$LLAMA_API_KEY" > "$KEYFILE")
[ -s "$KEYFILE" ] || (umask 077; openssl rand -hex 24 > "$KEYFILE")

loaded() { curl -s -H "Authorization: Bearer $(head -1 "$KEYFILE")" "localhost:$PORT/v1/models" 2>/dev/null | grep -o '"value": *"loaded"' | wc -l; }

start() {
  if pgrep -f "$BIN --models-preset" >/dev/null; then echo "already running"; return 0; fi
  setsid nohup "$BIN" --models-preset "$D/models.ini" --models-max 2 \
    --api-key-file "$KEYFILE" --host "$HOST" --port "$PORT" > "$LOG" 2>&1 < /dev/null &
  for i in $(seq 30); do
    curl -s -H "Authorization: Bearer $(head -1 "$KEYFILE")" "localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "ready: http://localhost:$PORT/v1  (models load on first request: gemma4-12b, gemma4-e4b)"; return 0; }
    pgrep -f "$BIN --models-preset" >/dev/null || { echo "failed to start, see $LOG"; return 1; }
    sleep 1
  done
  echo "timeout, see $LOG"; return 1
}
stop()   { pkill -f "$BIN --models-preset" && echo stopped || echo "not running"; sleep 3; }
unload() {
  for m in gemma4-12b gemma4-e4b; do
    curl -s -X POST "localhost:$PORT/models/unload" -H "Authorization: Bearer $(head -1 "$KEYFILE")" -H "Content-Type: application/json" -d "{\"model\": \"$m\"}" >/dev/null 2>&1 || true
  done
  echo "unloaded models"
}
status() {
  pgrep -f "$BIN --models-preset" >/dev/null && echo "router: running (:$PORT), loaded models: $(loaded)/2" || echo "router: stopped"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
}
case "$1" in
  start) start;; stop) stop;; restart) stop; start;; unload) unload;; status) status;;
  key) cat "$KEYFILE";;
  *) echo "usage: $0 {start|stop|restart|unload|status|key}"; exit 1;;
esac

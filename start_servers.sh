#!/usr/bin/env bash
# start_servers.sh — Launch all voice-agent servers.
#
# Servers started:
#   llm_server  → http://0.0.0.0:8000
#   tts_server  → http://0.0.0.0:8001
#   stt_server  → http://0.0.0.0:8002
#   agent_server→ http://0.0.0.0:8003
#
# Logs are written to logs/<server>.log
# PIDs are written to logs/<server>.pid
#
# Usage:
#   ./start_servers.sh          # start all servers
#   ./start_servers.sh stop     # stop all servers
#   ./start_servers.sh status   # show status of all servers
#   ./start_servers.sh restart  # stop then start all servers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Server definitions: name, module/script, port, extra args
# ---------------------------------------------------------------------------
declare -a SERVER_NAMES=(llm_server tts_server stt_server agent_server)

declare -A SERVER_CMD=(
    [llm_server]="python $SCRIPT_DIR/llm_server.py"
    [tts_server]="python $SCRIPT_DIR/tts_server.py --port 8001"
    [stt_server]="python $SCRIPT_DIR/stt_server.py --port 8002"
    [agent_server]="python $SCRIPT_DIR/agent_server.py --port 8003"
)

declare -A SERVER_PORT=(
    [llm_server]=8000
    [tts_server]=8001
    [stt_server]=8002
    [agent_server]=8003
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

pid_file()  { echo "$LOG_DIR/$1.pid"; }
log_file()  { echo "$LOG_DIR/$1.log"; }

is_running() {
    local name="$1"
    local pf; pf="$(pid_file "$name")"
    if [[ -f "$pf" ]]; then
        local pid; pid="$(cat "$pf")"
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

start_server() {
    local name="$1"
    local cmd="${SERVER_CMD[$name]}"
    local lf; lf="$(log_file "$name")"
    local pf; pf="$(pid_file "$name")"

    if is_running "$name"; then
        echo "  [$name] already running (pid $(cat "$pf"))"
        return
    fi

    echo "  [$name] starting on port ${SERVER_PORT[$name]} …"
    # shellcheck disable=SC2086
    nohup $cmd >> "$lf" 2>&1 &
    local pid=$!
    echo "$pid" > "$pf"
    echo "  [$name] pid=$pid  log=$lf"
}

stop_server() {
    local name="$1"
    local pf; pf="$(pid_file "$name")"

    if ! is_running "$name"; then
        echo "  [$name] not running"
        [[ -f "$pf" ]] && rm -f "$pf"
        return
    fi

    local pid; pid="$(cat "$pf")"
    echo "  [$name] stopping pid=$pid …"
    kill "$pid" 2>/dev/null || true

    # Wait up to 10 s for graceful shutdown
    local i=0
    while kill -0 "$pid" 2>/dev/null && (( i < 20 )); do
        sleep 0.5
        (( i++ ))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo "  [$name] force-killing pid=$pid"
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$pf"
    echo "  [$name] stopped"
}

status_server() {
    local name="$1"
    local pf; pf="$(pid_file "$name")"
    if is_running "$name"; then
        local pid; pid="$(cat "$pf")"
        printf "  %-15s \033[32mrunning\033[0m  pid=%-7s port=%s\n" \
            "$name" "$pid" "${SERVER_PORT[$name]}"
    else
        printf "  %-15s \033[31mstopped\033[0m\n" "$name"
    fi
}

wait_for_health() {
    local name="$1"
    local port="${SERVER_PORT[$name]}"
    local url="http://127.0.0.1:$port/health"
    local max_wait=120   # seconds
    local interval=2
    local elapsed=0

    echo "  [$name] waiting for health check at $url …"
    while (( elapsed < max_wait )); do
        if curl -sf "$url" -o /dev/null 2>/dev/null; then
            echo "  [$name] healthy after ${elapsed}s"
            return 0
        fi
        sleep "$interval"
        (( elapsed += interval ))
    done
    echo "  [$name] WARNING: did not become healthy within ${max_wait}s"
    return 1
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_start() {
    echo "=== Starting servers ==="
    # Start backends first, then agent
    for name in "${SERVER_NAMES[@]}"; do
        start_server "$name"
    done

    echo ""
    echo "=== Waiting for backends to become healthy ==="
    # Agent depends on the three backends; wait for them first
    for name in llm_server tts_server stt_server; do
        wait_for_health "$name" || true
    done

    echo ""
    echo "=== All servers launched ==="
    echo ""
    cmd_status
}

cmd_stop() {
    echo "=== Stopping servers ==="
    # Stop in reverse order
    for (( i=${#SERVER_NAMES[@]}-1; i>=0; i-- )); do
        stop_server "${SERVER_NAMES[$i]}"
    done
    echo "=== Done ==="
}

cmd_status() {
    echo "=== Server status ==="
    for name in "${SERVER_NAMES[@]}"; do
        status_server "$name"
    done
}

cmd_restart() {
    cmd_stop
    echo ""
    cmd_start
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${1:-start}" in
    start)   cmd_start   ;;
    stop)    cmd_stop    ;;
    status)  cmd_status  ;;
    restart) cmd_restart ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac

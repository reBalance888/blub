#!/bin/bash
# BLUB Ocean — One-click launcher
# Usage: bash start.sh [agent_count] [agent_type]
#   agent_count: number of agents (default: 20)
#   agent_type:  random|greedy|social|mix (default: mix)

set -e

AGENT_COUNT=${1:-20}
AGENT_TYPE=${2:-mix}
SERVER_PORT=8000

echo "=== BLUB Ocean ==="
echo "  Agents: $AGENT_COUNT ($AGENT_TYPE)"
echo "  Server: http://localhost:$SERVER_PORT"
echo "  Viewer: open viewer/index.html in browser"
echo ""

# Kill previous instances
echo "[1/3] Cleaning up old processes..."
pkill -f "python -m server.main" 2>/dev/null || true
pkill -f "agents/run_agents.py" 2>/dev/null || true
sleep 1

# Start server
echo "[2/3] Starting server..."
python -u -m server.main > server_log.txt 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
for i in $(seq 1 15); do
  if curl -s "http://localhost:$SERVER_PORT/stats" > /dev/null 2>&1; then
    echo "  Server ready (PID: $SERVER_PID)"
    break
  fi
  if [ $i -eq 15 ]; then
    echo "  ERROR: Server failed to start. Check server_log.txt"
    exit 1
  fi
  sleep 1
done

# Start agents
echo "[3/3] Launching $AGENT_COUNT $AGENT_TYPE agents..."
python -u agents/run_agents.py --count "$AGENT_COUNT" --type "$AGENT_TYPE" > agents_log.txt 2>&1 &
AGENTS_PID=$!
echo "  Agents running (PID: $AGENTS_PID)"

echo ""
echo "=== BLUB Ocean is running ==="
echo "  Server log: tail -f server_log.txt"
echo "  Agents log: tail -f agents_log.txt"
echo "  Epoch stats: grep EPOCH server_log.txt"
echo ""
echo "  Press Ctrl+C to stop everything"
echo ""

# Trap Ctrl+C to kill both
cleanup() {
  echo ""
  echo "Shutting down..."
  kill $AGENTS_PID 2>/dev/null || true
  kill $SERVER_PID 2>/dev/null || true
  echo "Done."
  exit 0
}
trap cleanup INT TERM

# Wait for either to exit
wait

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8100}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
export MODELS_DIR="$(cd .. && pwd)/models"

# Detect docker access (try sg docker wrapper, then direct, then sudo)
if sg docker -c "docker info" >/dev/null 2>&1; then
    docker_cmd() { sg docker -c "docker $*"; }
elif docker info >/dev/null 2>&1; then
    docker_cmd() { docker "$@"; }
elif sudo docker info >/dev/null 2>&1; then
    docker_cmd() { sudo docker "$@"; }
else
    echo "ERROR: Cannot access Docker." >&2
    echo "Fix: sudo usermod -aG docker \$USER && newgrp docker" >&2
    exit 1
fi
compose_cmd() { docker_cmd compose "$@"; }

# Check nvidia runtime
if ! docker_cmd info 2>&1 | grep -q nvidia; then
    echo "ERROR: NVIDIA Docker runtime not configured." >&2
    echo "" >&2
    echo "Run these commands (one-time setup):" >&2
    echo "  sudo nvidia-ctk runtime configure --runtime=docker" >&2
    echo "  sudo systemctl restart docker" >&2
    exit 1
fi

echo "=== Building Docker image (first build may take several minutes) ==="
compose_cmd build

echo "=== Starting container ==="
compose_cmd up -d

echo "=== Waiting for server (timeout: ${HEALTH_TIMEOUT}s) ==="
ELAPSED=0
while [ "$ELAPSED" -lt "$HEALTH_TIMEOUT" ]; do
    HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null || echo "000")

    if [ "$HEALTH_CODE" = "200" ]; then
        echo ""
        echo "=== Server ready! ==="
        echo ""
        echo "  API:     http://localhost:${PORT}"
        echo "  Health:  http://localhost:${PORT}/health"
        echo ""
        echo "  Quick test:"
        echo "  curl -X POST http://localhost:${PORT}/submit \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"prompt\": \"a red panda reading a book, watercolor\"}'"
        echo ""
        echo "  Logs:    sg docker -c 'docker compose logs -f zimage-server'"
        echo "  Stop:    bash stop.sh"
        exit 0
    fi

    if [ "$HEALTH_CODE" = "503" ]; then
        BODY=$(curl -s "http://localhost:${PORT}/health" 2>/dev/null || echo "")
        if echo "$BODY" | grep -q '"error"'; then
            echo ""
            echo "ERROR: Server startup failed!" >&2
            echo "$BODY" >&2
            echo "" >&2
            echo "Container logs:" >&2
            compose_cmd logs --tail 50 zimage-server >&2
            exit 1
        fi
    fi

    printf "."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

echo ""
echo "ERROR: Server did not become ready within ${HEALTH_TIMEOUT}s" >&2
echo "Container logs:" >&2
compose_cmd logs --tail 80 zimage-server >&2
exit 1

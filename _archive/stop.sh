#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect docker access
if sg docker -c "docker info" >/dev/null 2>&1; then
    docker_cmd() { sg docker -c "docker $*"; }
elif docker info >/dev/null 2>&1; then
    docker_cmd() { docker "$@"; }
elif sudo docker info >/dev/null 2>&1; then
    docker_cmd() { sudo docker "$@"; }
else
    echo "ERROR: Cannot access Docker." >&2
    exit 1
fi
compose_cmd() { docker_cmd compose "$@"; }

echo "=== Stopping Z-Image server ==="
compose_cmd down

echo "=== Stopped ==="

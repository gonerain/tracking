#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-tracking}"
IMAGE_NAME="${IMAGE_NAME:-tracking:noetic}"
MOUNT_TARGET="${MOUNT_TARGET:-/root/HDMap}"

run_shell='source /opt/ros/noetic/setup.bash && exec /bin/bash'

container_exists() {
  docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1
}

ensure_x11_access() {
  if [[ -n "${DISPLAY:-}" ]] && command -v xhost >/dev/null 2>&1; then
    xhost +SI:localuser:root >/dev/null 2>&1 || true
    xhost +local:docker >/dev/null 2>&1 || true
  fi
}

if container_exists; then
  docker container start "${CONTAINER_NAME}" >/dev/null
  exec docker exec -it "${CONTAINER_NAME}" /bin/bash -lc "${run_shell}"
fi

ensure_x11_access

docker_args=(
  -it
  --name "${CONTAINER_NAME}"
  -v "${WORKSPACE_ROOT}:${MOUNT_TARGET}"
)

if [[ -n "${DISPLAY:-}" ]]; then
  docker_args+=(
    -e "DISPLAY=${DISPLAY}"
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  )

  XAUTH="${XAUTHORITY:-$HOME/.Xauthority}"
  if [[ -f "${XAUTH}" ]]; then
    docker_args+=(
      -e "XAUTHORITY=/root/.Xauthority"
      -v "${XAUTH}:/root/.Xauthority:ro"
    )
  fi
fi

if [[ -n "${GDK_SCALE:-}" ]]; then
  docker_args+=(-e "GDK_SCALE=${GDK_SCALE}")
fi

if [[ -n "${GDK_DPI_SCALE:-}" ]]; then
  docker_args+=(-e "GDK_DPI_SCALE=${GDK_DPI_SCALE}")
fi

exec docker run "${docker_args[@]}" "${IMAGE_NAME}" /bin/bash -lc "${run_shell}"
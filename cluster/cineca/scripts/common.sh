#!/usr/bin/env bash
set -euo pipefail

CINECA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DIR="$CINECA_DIR/.local"
STEP_BIN="$LOCAL_DIR/bin/step"
SSH_CONFIG="$CINECA_DIR/ssh_config"
ENV_FILE="$CINECA_DIR/cluster.env"
AGENT_FILE="$LOCAL_DIR/ssh-agent.env"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

export STEPPATH="$LOCAL_DIR/step"
export PATH="$LOCAL_DIR/bin:$PATH"

CINECA_USER="${CINECA_USER:-}"
CINECA_HOST="${CINECA_HOST:-}"
SSH_ARGS=(-F "$SSH_CONFIG" -o "HostName=$CINECA_HOST" -o "User=$CINECA_USER")

load_agent() {
  if [[ -f "$AGENT_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$AGENT_FILE" >/dev/null
  fi
}

require_step() {
  if [[ ! -x "$STEP_BIN" ]]; then
    STEP_BIN="$(command -v step || true)"
  fi
  if [[ -z "$STEP_BIN" ]] || [[ ! -x "$STEP_BIN" ]]; then
      echo "Smallstep is not installed. Run brew install step first." >&2
      exit 1
  fi
}

require_env() {
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE. Copy cluster.env.example to cluster.env first." >&2
    exit 1
  fi
}

require_target() {
  require_env
  if [[ -z "$CINECA_USER" ]] || [[ -z "$CINECA_HOST" ]]; then
    echo "CINECA_USER/CINECA_HOST are unresolved. Inspect the certificate Principal and active Project Account first." >&2
    exit 2
  fi
}

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
require_step
require_env

if [[ -z "${CINECA_EMAIL:-}" ]]; then
  echo "Set CINECA_EMAIL in cluster.env. CINECA requires the registered email, not the cluster username." >&2
  exit 2
fi

mkdir -p "$LOCAL_DIR"
if [[ ! -f "$STEPPATH/config/defaults.json" ]]; then
  "$STEP_BIN" ca bootstrap \
    --ca-url=https://sshproxy.hpc.cineca.it \
    --fingerprint=2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
fi

load_agent
if [[ -z "${SSH_AUTH_SOCK:-}" ]] || ! ssh-add -l >/dev/null 2>&1; then
  eval "$(ssh-agent -s)" >/dev/null
  umask 077
  {
    printf 'export SSH_AUTH_SOCK=%q\n' "$SSH_AUTH_SOCK"
    printf 'export SSH_AGENT_PID=%q\n' "$SSH_AGENT_PID"
  } > "$AGENT_FILE"
fi

echo "A browser window will request the CINECA password and OTP. Nothing is stored in this repository."
"$STEP_BIN" ssh login "$CINECA_EMAIL" --provisioner cineca-hpc
"$STEP_BIN" ssh list


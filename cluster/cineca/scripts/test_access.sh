#!/usr/bin/env bash
set -u

username="${CINECA_TEST_USER:-bliu0001}"
known_hosts="$(mktemp)"
trap 'rm -f "$known_hosts"' EXIT

scan_cluster() {
  local cluster="$1"
  shift
  local address keyal
  for keyal in rsa ecdsa ed25519; do
    for address in "$@"; do
      ssh-keyscan -T 10 -t "$keyal" "$address" 2>/dev/null |
        sed "s/$address/login*.$cluster.cineca.it/g" >> "$known_hosts"
    done
  done
}

scan_cluster leonardo \
  login01-ext.leonardo.cineca.it \
  login02-ext.leonardo.cineca.it \
  login05-ext.leonardo.cineca.it \
  login07-ext.leonardo.cineca.it

scan_cluster pitagora \
  login01-ext.pitagora.cineca.it \
  login02-ext.pitagora.cineca.it \
  login03-ext.pitagora.cineca.it \
  login04-ext.pitagora.cineca.it \
  login05-ext.pitagora.cineca.it \
  login06-ext.pitagora.cineca.it

if [[ ! -s "$known_hosts" ]]; then
  echo "Unable to retrieve CINECA host keys" >&2
  exit 1
fi

connected=0
for cluster in leonardo pitagora; do
  echo "===== testing $cluster ====="
  if ssh \
    -o BatchMode=yes \
    -o ConnectTimeout=15 \
    -o StrictHostKeyChecking=yes \
    -o "UserKnownHostsFile=$known_hosts" \
    "$username@login.$cluster.cineca.it" \
    'echo CONNECTED; whoami; hostname; command -v saldo >/dev/null && saldo -b; echo "== paths =="; printf "HOME=%s\nWORK=%s\nFAST=%s\nSCRATCH=%s\n" "$HOME" "${WORK:-}" "${FAST:-}" "${SCRATCH:-}"; echo "== conda =="; command -v conda || true; type module >/dev/null 2>&1 && module -t avail 2>&1 | grep -Ei "conda|anaconda|miniconda" | head -n 20 || true'; then
    connected=1
    break
  fi
done

exit $((connected == 1 ? 0 : 1))

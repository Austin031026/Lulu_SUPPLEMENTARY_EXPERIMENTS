#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

VERSION="${STEP_VERSION:-0.30.6}"
ARCH="$(uname -m)"
case "$ARCH" in
  arm64) ASSET_ARCH=arm64 ;;
  x86_64) ASSET_ARCH=amd64 ;;
  *) echo "Unsupported macOS architecture: $ARCH" >&2; exit 1 ;;
esac

ASSET="step_darwin_${VERSION}_${ASSET_ARCH}.tar.gz"
BASE_URL="https://dl.smallstep.com/gh-release/cli/gh-release-header/v${VERSION}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$LOCAL_DIR/bin"
ASSET_SIZE="${STEP_ASSET_SIZE:-14017362}"
CHUNKS="${STEP_DOWNLOAD_CHUNKS:-16}"
CHUNK_SIZE=$(( (ASSET_SIZE + CHUNKS - 1) / CHUNKS ))
pids=()
for ((i=0; i<CHUNKS; i++)); do
  start=$((i * CHUNK_SIZE))
  end=$((start + CHUNK_SIZE - 1))
  (( end >= ASSET_SIZE )) && end=$((ASSET_SIZE - 1))
  curl -fsSL --retry 3 --connect-timeout 20 --max-time 300 \
    --range "$start-$end" "$BASE_URL/$ASSET" -o "$TMP_DIR/part.$i" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

dd if=/dev/zero of="$TMP_DIR/$ASSET" bs=1 count=0 seek="$ASSET_SIZE" 2>/dev/null
for ((i=0; i<CHUNKS; i++)); do
  start=$((i * CHUNK_SIZE))
  dd if="$TMP_DIR/part.$i" of="$TMP_DIR/$ASSET" bs=1 seek="$start" conv=notrunc 2>/dev/null
done
curl -fsSL --retry 3 --connect-timeout 20 --max-time 60 "$BASE_URL/checksums.txt" -o "$TMP_DIR/checksums.txt"

EXPECTED="$(awk -v asset="$ASSET" '$2 == asset {print $1}' "$TMP_DIR/checksums.txt")"
if [[ -z "$EXPECTED" ]]; then
  echo "No checksum found for $ASSET" >&2
  exit 1
fi
ACTUAL="$(shasum -a 256 "$TMP_DIR/$ASSET" | awk '{print $1}')"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo "Checksum mismatch for $ASSET" >&2
  exit 1
fi

tar -xzf "$TMP_DIR/$ASSET" -C "$TMP_DIR"
FOUND="$(find "$TMP_DIR" -type f -name step -perm -u+x | head -n 1)"
if [[ -z "$FOUND" ]]; then
  echo "The archive did not contain an executable named step" >&2
  exit 1
fi
install -m 0755 "$FOUND" "$STEP_BIN"
"$STEP_BIN" version

#!/usr/bin/env bash
set -euo pipefail

REPO="ankitjk/RunItBack"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/models"

mkdir -p "$DEST"

echo "Downloading scene classifier model from latest release..."

curl -fSL "https://github.com/$REPO/releases/latest/download/scene_classifier.onnx" \
  -o "$DEST/scene_classifier.onnx"

curl -fSL "https://github.com/$REPO/releases/latest/download/scene_classifier.onnx.data" \
  -o "$DEST/scene_classifier.onnx.data"

echo "Model saved to $DEST"

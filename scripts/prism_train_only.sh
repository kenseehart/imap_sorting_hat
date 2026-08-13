#!/bin/bash
set -euo pipefail
cd /home/mcp/mcp-gateway
. .venv/bin/activate
set -a
. /home/mcp/.config/fish/fish.env
set +a

echo "=== corpus stats ==="
python -m fish corpus stats --json | head -c 2000
echo

echo "=== ensure torch ==="
python -c 'import torch; print(torch.__version__)' 2>/dev/null || pip install 'torch>=2.0.0' --index-url https://download.pytorch.org/whl/cpu

echo "=== prism-train (batched) ==="
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p /data/fish/models
python -m fish prism-train --epochs 5 --json --output /data/fish/models/personal.prz
ls -la /data/fish/models/personal.prz

echo "=== set FISH_PRISM_MODEL ==="
ENVF=/home/mcp/.config/fish/fish.env
if grep -q '^FISH_PRISM_MODEL=' "$ENVF" 2>/dev/null; then
  sed -i 's|^FISH_PRISM_MODEL=.*|FISH_PRISM_MODEL=personal.prz|' "$ENVF"
else
  printf '\nFISH_PRISM_MODEL=personal.prz\n' >> "$ENVF"
fi
grep '^FISH_PRISM_MODEL=' "$ENVF"

echo "=== restart mcp-fish ==="
sudo systemctl restart mcp-fish
sleep 3
systemctl is-active mcp-fish

echo "=== TRAIN_DEPLOY_OK ==="

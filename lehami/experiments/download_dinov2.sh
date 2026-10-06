#!/usr/bin/env bash
# Download the frozen DINOv2 encoder (dinov2_vits14, ~89 MB) into the torch.hub
# cache. Needed only by conservative_dino.sh; run it once, on a machine with
# network access. The encoder refuses to build if this cache is missing, so a
# node without network fails fast instead of hanging.
#
# Usage:
#   export TORCH_HOME=/path/to/torch_home   # optional, default ~/.cache/torch
#   bash lehami/experiments/download_dinov2.sh
#
# Use the same TORCH_HOME when you later run conservative_dino.sh.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
mkdir -p "$TORCH_HOME"
log "TORCH_HOME=$TORCH_HOME"

"${PY[@]}" - <<'PY'
import torch

# Skip torch.hub's fork check: it calls the rate-limited GitHub API for nothing.
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
print(f"loaded dinov2_vits14: num_features={m.num_features} patch_size={m.patch_size}")
# A second load must be served from the cache without touching the network.
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
print("second load served from cache OK")
PY
du -sh "$TORCH_HOME"

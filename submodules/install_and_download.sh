#!/bin/bash
# Run from repo root: bash submodules/install_and_download.sh
# Requires: conda activate ttocc

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> 3DGS CUDA extensions"
if [ ! -d submodules/simple-knn ]; then
  git clone --depth 1 https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
fi
touch submodules/simple-knn/simple_knn/__init__.py

pip uninstall -y simple-knn simple_kNN 2>/dev/null || true
pip install --no-build-isolation -e ./submodules/simple-knn
pip install --no-build-isolation -e ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation -e ./submodules/diff-gaussian-rasterization_semantic

echo "==> VGGT & RAFT"
pip install -e ./submodules/VGGT
pip install -e ./submodules/RAFT

echo "==> OpenSeeD (unified in ttocc; detectron2 fork + deformable ops)"
pip install 'transformers>=4.30,<4.46'
pip install --no-build-isolation 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install --no-build-isolation 'git+https://github.com/cocodataset/panopticapi.git'
pip install --no-build-isolation -e ./submodules/OpenSeeD/openseed/body/encoder/ops

echo "==> Download checkpoints"
cd submodules/OpenSeeD
if [ ! -f model_state_dict_swint_51.2ap.pt ]; then
  wget -c https://github.com/IDEA-Research/OpenSeeD/releases/download/openseed/model_state_dict_swint_51.2ap.pt
fi
echo "OpenSeeD weights OK"

cd "$ROOT/submodules/RAFT"
if [ ! -f raft-things.pth ]; then
  gdown --id 1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM -O raft-things.pth
fi
echo "RAFT weights OK"

cd "$ROOT"
if [ ! -f submodules/RAFT/raft-things.pth ]; then
  ln -s "submodules/RAFT/raft-things.pth" "raft-things.pth"
fi

echo "Done. All components install into the active ttocc environment."

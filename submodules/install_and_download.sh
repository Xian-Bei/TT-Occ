#!/bin/bash
# Run from repo root: bash submodules/install_and_download.sh
# Requires: conda activate ttocc

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Shared runtime pins"
pip install "huggingface_hub==0.36.2" "numpy==1.26.4" "pillow==10.4.0"

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

echo "==> Rex-Omni + SAM"
pip install qwen_vl_utils transformers==4.51.3 accelerate==1.10.1
pip install pycocotools==2.0.10 shapely==2.1.2
pip install -e ./submodules/Rex-Omni --no-deps
pip install git+https://github.com/facebookresearch/segment-anything.git

echo "==> MapAnything + geometry/tracking deps"
pip install -e ./submodules/map-anything
pip install pycolmap==3.10.0
pip install git+https://github.com/cvg/LightGlue.git

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

mkdir -p "$ROOT/submodules/Rex-Omni/checkpoints"
if [ ! -f "$ROOT/submodules/Rex-Omni/checkpoints/sam_vit_h_4b8939.pth" ]; then
  wget -c -O "$ROOT/submodules/Rex-Omni/checkpoints/sam_vit_h_4b8939.pth" \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi
echo "Rex-Omni SAM weights OK"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  pip install "huggingface_hub==0.36.2"
fi
huggingface-cli download IDEA-Research/Rex-Omni \
  --resume-download \
  --local-dir /home/$USER/.cache/huggingface/hub/models--IDEA-Research--Rex-Omni
echo "Rex-Omni HF weights OK"

echo "==> Optional speed-up (FlashAttention2 for Rex-Omni)"
echo "    FLASH_ATTN_CUDA_ARCHS=\"89\" MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation"

echo "Done. All components install into the active ttocc environment."

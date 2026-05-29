#!/bin/bash

set -e

if [ ! -d submodules/simple-knn ]; then
  git clone --depth 1 https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
fi
touch submodules/simple-knn/simple_knn/__init__.py

pip uninstall -y simple-knn simple_kNN 2>/dev/null || true
pip install --no-build-isolation -e ./submodules/simple-knn
pip install --no-build-isolation -e ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation -e ./submodules/diff-gaussian-rasterization_semantic
pip install -e ./submodules/VGGT
pip install -e ./submodules/RAFT

cd submodules/OpenSeeD
wget -c https://github.com/IDEA-Research/OpenSeeD/releases/download/openseed/model_state_dict_swint_51.2ap.pt
echo "OpenSeeD model downloaded"

cd ../RAFT
gdown --id 1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM -O raft-things.pth
echo "raft model downloaded"


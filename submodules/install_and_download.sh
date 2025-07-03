#!/bin/bash

set -e  
pip install -e ./submodules/VGGT
pip install -e ./submodules/RAFT

cd submodules/OpenSeeD
wget -c https://github.com/IDEA-Research/OpenSeeD/releases/download/openseed/model_state_dict_swint_51.2ap.pt
echo "OpenSeeD model downloaded"

cd ../RAFT
gdown --id 1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM -O raft-things.pth
echo "raft model downloaded"


# TT-Occ: Test-Time Compute for Self-Supervised Occupancy
## Under Construction
![demo](assets/teaser.gif)
**Why train a dedicated occupancy network in the era of foundation VLMs?**
We show that a test-time occupancy framework that integreted with a conmbination of VLMs could achieve SOTA performance **without any network training or fine-tuning**.

> TT-Occ easily integrates advanced VLMs through a simple adapter interface. Contributions are warmly welcomed!

## 📥 Data Preparation

1. Download the nuScenes dataset from [nuScenes.org](https://www.nuscenes.org/download).
2. Extract nuScenes data into a readable format:
   ```bash
   python extract_nuscenes.py  # Ensure you set your dataset path inside this script.
   ```
3. Download ground-truth occupancy labels:
   * Occ3D-nuScenes GT: [Google Drive](https://drive.google.com/drive/folders/1Xarc91cNCNN3h8Vum-REbI-f0UlSf5Fc)
   * nuCraft GT: [nuCraft GitHub](https://github.com/V2AI/nuCraft_API?tab=readme-ov-file)
4. Update `data_root` of all scripts in ``.

## 🌱 Environment Setup
For the external repositories used in this project, we provide minimal versions of their codebases under the `submodules` directory (with their original licenses retained; please respect and comply with their terms). These have been packaged under a unified conda environment, so you do not need to clone each dependency separately.

To reproduce our environment reliably, use:

```bash
conda env create -f environment.yaml
conda activate ttocc
bash submodules/install_and_download.sh
```

`install_and_download.sh` (run from the repo root, with `ttocc` activated) includes all install and checkpoint download steps required by TT-Occ (OpenSeeD / Rex-Omni / VGGT / MapAnything / RAFT / 3DGS CUDA extensions).

## ✅ Currently Supported Online Models

TT-Occ currently supports the following online test-time providers:

- Semantic models: `openseed`, `rexomni`
- Depth models: `vggt`, `mapanything`
- Dynamic mask model: `raft`

Selection is controlled by environment variables in `run_main.sh`:

- `SEMANTIC_PREFIX` (`openseed` by default)
- `DEPTH_PREFIX` (`vggt` by default)
- `DYNAMIC_MASK_PREFIX` (`raft` by default)

`install_and_download.sh` will:

- Clone `simple-knn` if missing, then install **3DGS CUDA extensions** (`simple-knn`, `diff-gaussian-rasterization`, `diff-gaussian-rasterization_semantic`). Do **not** use `pip install simple-knn` from PyPI — that is a different package.
- `pip install -e` **VGGT** and **RAFT**
- Install **OpenSeeD** into the same `ttocc` env: `detectron2` (IDEA fork), `panopticapi`, deformable-attention CUDA ops, and `transformers` for CLIP (`OpenSeeD/main.py` includes a Pillow≥10 compat shim for detectron2)
- Download OpenSeeD and RAFT checkpoint weights under `submodules/`


```
# Rex-Omni minimal runtime deps (transformers backend)
pip install qwen_vl_utils transformers==4.51.3 accelerate==1.10.1
pip install pycocotools==2.0.10 shapely==2.1.2
pip install -e "./submodules/Rex-Omni" --no-deps
pip install git+https://github.com/facebookresearch/segment-anything.git

# MapAnything + COLMAP + tracking deps used by ttocc.py
pip install -e "./submodules/map-anything"
pip install pycolmap==3.10.0
pip install git+https://github.com/cvg/LightGlue.git

# Shared versions to avoid cross-package breakage
pip install "numpy==1.26.4" "pillow==10.4.0"
```

Download checkpoints:

```bash
# SAM checkpoint for rexsam_ttocc.py
cd submodules/Rex-Omni
mkdir -p checkpoints
wget -c -O checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Rex-Omni model weights (HuggingFace)
huggingface-cli download IDEA-Research/Rex-Omni \
  --resume-download \
  --local-dir /home/$USER/.cache/huggingface/hub/models--IDEA-Research--Rex-Omni
```


Notes:

- `Rex-Omni` may try to use FlashAttention2 by default. In this repo, a fallback to `eager` attention is enabled when `flash_attn` is unavailable.
- Keep `huggingface_hub` `<1.0` with `transformers==4.51.3` (use `0.36.2`).
- `map-anything/ttocc.py` assumes 6 cameras and writes outputs to `mapanything3_{cam}`.


## 🚀 Running TT-Occ

Evaluate on the complete 150-scene test split:

```bash
conda activate ttocc
bash run_main.sh
```

By default `run_main.sh` enables **mIoU** (`EVAL_OCC=1`) and writes per-scene `result.json` under `out-main-Occ3D/<variant>/<scene>/`.

Manual flags for `train.py`:

- `--use_fusion` — enable E-style semantic/radiometric fusion (default off, D-equivalent)
- `--eval_occ --occ3d_path ... --nucraft_path ...` — mIoU vs saved `Occ/*.pth`

Offline aggregation (same logic as train): `python summarymiou.py`, `python summary.py`.

### 🎨 Visualization

We provide a simple occupancy visualizer based on **Open3D**.
To visualize occupancy predictions, run:

```bash
python vis.py  # Make sure the dataset path is correctly set.
```

Example visualization outputs:

* **TT-OccLiDAR:**
  ![TT-OccLiDAR](assets/scene-0039_15_1.png)

* **TT-OccCamera:**
  ![TT-OccCamera](assets/scene-0039_15_0.png)

For advanced visualization commands, refer to `custom_utils/VoxelGridVisualizer`.


## 📌 Acknowledgements

This project builds upon the excellent codebase of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and powerful VLMs including [OpenSeeD](https://github.com/IDEA-Research/OpenSeeD), [Rex-Omni](https://github.com/IDEA-Research/Rex-Omni), [VGGT](https://github.com/facebookresearch/vggt), [MapAnything](https://github.com/facebookresearch/map-anything), [RAFT](https://github.com/princeton-vl/RAFT), and [TT-Occ](./).
We deeply appreciate their creators' efforts and your interest in TT-Occ!

## 📖 Citation

If you find this work helpful, please star our repo and cite the paper:

```bibtex
@InProceedings{ttocc,
    author    = {Zhang, Fengyi and Sun, Xiangyu and Yang, Huitong and Zhang, Zheng and Huang, Zi and Luo, Yadan},
    title     = {Test-Time 3D Occupancy Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {35691-35701}
}
```
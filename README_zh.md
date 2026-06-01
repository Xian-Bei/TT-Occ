# TT-Occ: 面向自监督占据预测的测试时计算框架
## 持续更新中
![demo](assets/teaser.gif)

**在基础模型时代，为什么还要单独训练占据网络？**  
我们展示了一个测试时占据预测框架：通过组合多个 VLM，无需训练或微调即可达到 SOTA 表现。

> TT-Occ 通过统一适配接口支持多种先进 VLM，欢迎贡献！

## 📥 数据准备

1. 从 [nuScenes.org](https://www.nuscenes.org/download) 下载 nuScenes 数据集。
2. 将 nuScenes 提取为本项目可读格式：
   ```bash
   python extract_nuscenes.py  # 请先在脚本内设置数据集路径
   ```
3. 下载占据真值标签：
   - Occ3D-nuScenes GT: [Google Drive](https://drive.google.com/drive/folders/1Xarc91cNCNN3h8Vum-REbI-f0UlSf5Fc)
   - nuCraft GT: [nuCraft GitHub](https://github.com/V2AI/nuCraft_API?tab=readme-ov-file)
4. 按需更新脚本中的 `data_root` 路径。

## 🌱 环境安装

项目依赖的外部代码均提供在 `submodules` 下（保留原始许可证，请遵守相应条款），并统一到同一个 conda 环境中。

建议按下面流程安装：

```bash
conda env create -f environment.yaml
conda activate ttocc
bash submodules/install_and_download.sh
```

`submodules/install_and_download.sh`（在仓库根目录执行，且已激活 `ttocc`）已包含完整流程：安装依赖并下载所需 checkpoint（OpenSeeD / Rex-Omni / VGGT / MapAnything / RAFT / 3DGS CUDA 扩展）。

## ✅ 当前已支持的在线模型

TT-Occ 当前支持以下在线 test-time provider：

- 语义模型：`openseed`, `rexomni`
- 深度模型：`vggt`, `mapanything`
- 动态掩码模型：`raft`

在 `run_main.sh` 中通过环境变量选择：

- `SEMANTIC_PREFIX`（默认 `openseed`）
- `DEPTH_PREFIX`（默认 `vggt`）
- `DYNAMIC_MASK_PREFIX`（默认 `raft`）

## 🚀 运行 TT-Occ

在 150 个测试场景上评估：

```bash
conda activate ttocc
bash run_main.sh
```

默认 `run_main.sh` 启用 **mIoU**（`EVAL_OCC=1`），并在每个场景目录下保存 `result.json`。  

`train.py` 常用参数：

- `--use_fusion`：启用 E-style 语义/辐射融合（默认关闭，等价 D 配置）
- `--eval_occ --occ3d_path ... --nucraft_path ...`：与保存的 `Occ/*.pth` 计算 mIoU

离线汇总脚本：`python summarymiou.py`, `python summary.py`

### 🎨 可视化

项目提供基于 **Open3D** 的占据可视化：

```bash
python vis.py  # 请确保脚本中的数据路径已正确设置
```

示例效果：

- **TT-OccLiDAR:**
  ![TT-OccLiDAR](assets/scene-0039_15_1.png)

- **TT-OccCamera:**
  ![TT-OccCamera](assets/scene-0039_15_0.png)

更多可视化能力请参考 `custom_utils/VoxelGridVisualizer`。

## 📌 致谢

本项目基于 [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) 以及多种优秀 VLM，包括 [OpenSeeD](https://github.com/IDEA-Research/OpenSeeD)、[Rex-Omni](https://github.com/IDEA-Research/Rex-Omni)、[VGGT](https://github.com/facebookresearch/vggt)、[MapAnything](https://github.com/facebookresearch/map-anything)、[RAFT](https://github.com/princeton-vl/RAFT) 和 [TT-Occ](./)。  
感谢原作者团队，也感谢你对 TT-Occ 的关注！

## 📖 引用

如果该工作对你有帮助，欢迎 star 并引用：

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

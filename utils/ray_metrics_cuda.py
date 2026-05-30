# Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
# Modified by Haisong Liu
import math
import copy
import ipdb
import numpy as np
import torch
from torch.utils.cpp_extension import load
from tqdm import tqdm
from prettytable import PrettyTable


dvr = load("dvr", sources=["utils/dvr/dvr.cpp", "utils/dvr/dvr.cu"], verbose=False, extra_cuda_cflags=['-allow-unsupported-compiler'])
_pc_range = [-40, -40, -1.0, 40, 40, 5.4]
_voxel_size = 0.4


# https://github.com/tarashakhurana/4d-occ-forecasting/blob/ff986082cd6ea10e67ab7839bf0e654736b3f4e2/test_fgbg.py#L29C1-L46C16
def get_rendered_pcds(origin, points, tindex, pred_dist):
    pcds = []
    
    for t in range(len(origin)):
        mask = (tindex == t)
        # skip the ones with no data
        if not mask.any():
            continue
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
        pcds.append(torch.from_numpy(pred_pts))
        
    return pcds


def meshgrid3d(occ_size, pc_range):
    W, H, D = occ_size
    
    xs = torch.linspace(0.5, W - 0.5, W).view(W, 1, 1).expand(W, H, D) / W
    ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(W, H, D) / H
    zs = torch.linspace(0.5, D - 0.5, D).view(1, 1, D).expand(W, H, D) / D
    xs = xs * (pc_range[3] - pc_range[0]) + pc_range[0]
    ys = ys * (pc_range[4] - pc_range[1]) + pc_range[1]
    zs = zs * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack((xs, ys, zs), -1)

    return xyz


def generate_lidar_rays():
    # prepare lidar ray angles
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)
    
    # nuscenes lidar fov: [0.2107773983152201, -0.5439104895672159] (rad)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    lidar_rays = []
    for pitch_angle in pitch_angles:
        for azimuth_angle in np.arange(0, 360, 1):
            azimuth_angle = np.deg2rad(azimuth_angle)

            x = np.cos(pitch_angle) * np.cos(azimuth_angle)
            y = np.cos(pitch_angle) * np.sin(azimuth_angle)
            z = np.sin(pitch_angle)

            lidar_rays.append((x, y, z))

    return np.array(lidar_rays, dtype=np.float32)


import torch
import copy

def process_one_sample(
    sem_pred,
    lidar_rays,
    output_origin,
    instance_pred=None,
    occ_class_names=None,
):
    """
    Args:
        sem_pred: (H, W, Z) long tensor, semantic prediction.
        lidar_rays: (N, 3) float tensor, direction vectors from lidar origin.
        output_origin: (1, T, 3) float tensor, lidar origin positions for each time step.
        instance_pred: (H, W, Z) long tensor, instance IDs (optional).
        occ_class_names: list of class names, last one assumed to be 'free' class.
        _pc_range: (6,) list/array, point cloud range [xmin, ymin, zmin, xmax, ymax, zmax].
        _voxel_size: float, voxel resolution.
        dvr: module providing render_forward().

    Returns:
        pred_pcds_t: (T*N, C) float tensor in CUDA.
                     C = 2 (label, dist) if no instance_pred, else 3 (label, instance, dist).
    """
    device = sem_pred.device
    T = output_origin.shape[1]
    pred_pcds_t = []

    # Prepare occupancy prediction (binary: 1=occupied, 0=free)
    free_id = len(occ_class_names) - 1
    occ_pred = copy.deepcopy(sem_pred)
    occ_pred[sem_pred < free_id] = 1
    occ_pred[sem_pred == free_id] = 0
    occ_pred = occ_pred.permute(2, 1, 0)  # (Z, W, H)
    occ_pred = occ_pred[None, None, :].contiguous().float().to(device)  # (1, 1, Z, W, H)

    offset = torch.tensor(_pc_range[:3], device=device).view(1, 1, 3)
    scaler = torch.tensor([_voxel_size] * 3, device=device).view(1, 1, 3)

    lidar_tindex = torch.zeros((1, lidar_rays.shape[0]), device=device)

    for t in range(T):
        lidar_origin = output_origin[:, t:t+1, :]  # (1, 1, 3)
        lidar_endpts = lidar_rays[None] + lidar_origin  # (1, N, 3)

        # Normalize to voxel coordinates
        output_origin_render = ((lidar_origin - offset) / scaler).float()  # (1, 1, 3)
        output_points_render = ((lidar_endpts - offset) / scaler).float()  # (1, N, 3)
        output_tindex_render = lidar_tindex  # (1, N)

        with torch.no_grad():
            pred_dist, _, coord_index = dvr.render_forward(
                occ_pred,
                output_origin_render,
                output_points_render,
                output_tindex_render,
                [1, 16, 200, 200],
                "test"
            )
            pred_dist *= _voxel_size  # still tensor in CUDA

        # coord_index: (1, N, 3)
        coord_index = coord_index[0, :, :].long()  # (N, 3)

        pred_label = sem_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][:, None].float()  # (N, 1)
        pred_dist = pred_dist[0, :, None]  # (N, 1)

        if instance_pred is not None:
            pred_instance = instance_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][:, None].float()
            pred_pcds = torch.cat([pred_label, pred_instance, pred_dist], dim=-1)  # (N, 3)
        else:
            pred_pcds = torch.cat([pred_label, pred_dist], dim=-1)  # (N, 2)

        pred_pcds_t.append(pred_pcds)

    pred_pcds_t = torch.cat(pred_pcds_t, dim=0)  # (T*N, C)
    return pred_pcds_t  # CUDA tensor


@torch.no_grad()
def calc_rayiou(pcd_pred_list, pcd_gt_list, occ_class_names):
    """
    Args:
        pcd_pred_list: list[Tensor], each (N, 2/3...) on CUDA.
                       [:,0]=pred_class_id (int/float), [:,1]=pred_depth (float, meters)
        pcd_gt_list:   list[Tensor], same shape as pred; [:,0]=gt_class_id, [:,1]=gt_depth
        occ_class_names: list[str], 最后一个是 free/empty 类
    Returns:
        iou_list: list[Tensor], len=3 for thresholds {1,2,4}m,
                  each is shape (num_classes-1,) on CUDA
    """
    device = pcd_pred_list[0].device
    C = len(occ_class_names)
    thresholds = torch.tensor([1.0, 2.0, 4.0], device=device)

    gt_cnt  = torch.zeros(C, device=device, dtype=torch.float32)
    pred_cnt= torch.zeros(C, device=device, dtype=torch.float32)
    tp_cnt  = torch.zeros(3, C, device=device, dtype=torch.float32)  # (T, C)

    for pcd_pred, pcd_gt in zip(pcd_pred_list, pcd_gt_list):
        # 切片 & 类型
        pred_label = pcd_pred[:, 0].long()
        gt_label   = pcd_gt[:, 0].long()
        depth_pred = pcd_pred[:, 1].float()
        depth_gt   = pcd_gt[:, 1].float()

        # 累加每类的预测/GT 数量
        gt_cnt  += torch.bincount(gt_label, minlength=C).float()
        pred_cnt+= torch.bincount(pred_label, minlength=C).float()

        # L1 深度误差
        l1 = (depth_pred - depth_gt).abs()

        # 三个阈值下的 TP（类别一致 + 深度误差小于阈值）
        same_cls = (pred_label == gt_label)
        for ti, thr in enumerate(thresholds):
            mask = same_cls & (l1 < thr)
            # 被命中的类别（用 gt_label 或 pred_label 都一样，因为 same_cls=True）
            cls_hit = gt_label[mask]
            if cls_hit.numel() > 0:
                tp_cnt[ti] += torch.bincount(cls_hit, minlength=C).float()

    # IoU = TP / (GT + Pred - TP)，去掉最后一个 free 类
    iou_list = []
    denom = (gt_cnt + pred_cnt - tp_cnt)  # (T, C) 广播：gt_cnt/pred_cnt 会自动expand
    # 防 0：置为 NaN 的地方
    for ti in range(3):
        numer = tp_cnt[ti]                  # (C,)
        d = denom[ti]                       # (C,)
        iou = numer / d.clamp(min=0)        # 先除，后面把 0 处设 NaN
        iou = torch.where(d > 0, iou, torch.nan)
        iou_list.append(iou[:-1])           # 去掉 free 类

    return iou_list


@torch.no_grad()
def main_rayiou(sem_pred_list, sem_gt_list, lidar_origin_list, occ_class_names, shape=(200, 200, 16)):
    """
    全 CUDA 版本的 RayIoU 评测。
    依赖：
      - process_one_sample(...) 返回 CUDA tensor，shape (T*N, 2) 或 (T*N, 3)，其中 [:,0]=class_id, [:,1]=depth
      - calc_rayiou_gpu(...) 返回长度=3的列表，每个元素是 (num_classes-1,) 的 CUDA tensor（去掉 free 类）
    """

    device = sem_pred_list[0].device

    # 生成 lidar 射线（转到 CUDA）
    lidar_rays = torch.from_numpy(generate_lidar_rays()).to(device=device, dtype=torch.float32)

    pcd_pred_list, pcd_gt_list = [], []

    free_id = len(occ_class_names) - 1

    for sem_pred, sem_gt, lidar_origins in tqdm(zip(sem_pred_list, sem_gt_list, lidar_origin_list), ncols=50):
        # 若输入不是期望形状，可视需要 reshape
        # sem_pred = sem_pred.view(*shape)
        # sem_gt   = sem_gt.view(*shape)

        # 生成 (ray, info) 的列表：[:,0]=label, [:,1]=depth
        pcd_pred = process_one_sample(
            sem_pred=sem_pred,
            lidar_rays=lidar_rays,
            output_origin=lidar_origins,
            occ_class_names=occ_class_names
        )
        pcd_gt = process_one_sample(
            sem_pred=sem_gt,
            lidar_rays=lidar_rays,
            output_origin=lidar_origins,
            occ_class_names=occ_class_names
        )

        # 只在非 free 类的射线上评测
        # 注意：我们在 CUDA 上做 mask
        valid_mask = (pcd_gt[:, 0].long() != free_id)
        pcd_pred = pcd_pred[valid_mask]
        pcd_gt   = pcd_gt[valid_mask]

        assert pcd_pred.shape == pcd_gt.shape
        pcd_pred_list.append(pcd_pred)
        pcd_gt_list.append(pcd_gt)

    # 计算 RayIoU（三个阈值）
    iou_list = calc_rayiou(pcd_pred_list, pcd_gt_list, occ_class_names)  # list of 3 tensors, each (C-1,)

    # MEAN 聚合
    rayiou_0 = torch.nanmean(iou_list[0])  # RayIoU@1
    rayiou_1 = torch.nanmean(iou_list[1])  # RayIoU@2
    rayiou_2 = torch.nanmean(iou_list[2])  # RayIoU@4
    # overall 平均（对三种阈值与所有类别做整体均值）
    rayiou   = torch.nanmean(torch.stack([iou_list[0], iou_list[1], iou_list[2]], dim=0))

    # ------- 打印表格（转到 CPU/float）-------
    table = PrettyTable(['Class Names', 'RayIoU@1', 'RayIoU@2', 'RayIoU@4'])
    table.float_format = '.3'

    for i in range(len(occ_class_names) - 1):
        r1 = iou_list[0][i].item() if torch.isfinite(iou_list[0][i]) else float('nan')
        r2 = iou_list[1][i].item() if torch.isfinite(iou_list[1][i]) else float('nan')
        r4 = iou_list[2][i].item() if torch.isfinite(iou_list[2][i]) else float('nan')
        table.add_row([occ_class_names[i], r1, r2, r4], divider=(i == len(occ_class_names) - 2))

    table.add_row(['MEAN', rayiou_0.item(), rayiou_1.item(), rayiou_2.item()])
    print(table)
    # ---------------------------------------

    return {
        'RayIoU':  rayiou.item(),
        'RayIoU@1': rayiou_0.item(),
        'RayIoU@2': rayiou_1.item(),
        'RayIoU@4': rayiou_2.item(),
    }


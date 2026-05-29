import argparse
import json
import os
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Tuple

import numpy as np
from pycocotools import mask as coco_mask
from shapely.geometry import Polygon
from tqdm import tqdm


def decode_rle(rle_str, size):
    """Decode RLE (Run Length Encoding) string to binary mask using pycocotools"""
    try:
        # Create RLE dict format expected by pycocotools
        rle = {"counts": rle_str.encode("utf-8"), "size": size}

        # Use pycocotools to decode
        mask = coco_mask.decode(rle)
        return mask

    except Exception as e:
        print(f"Error decoding RLE: {e}, RLE string: {rle_str[:50]}...")
        # Return empty mask if decoding fails
        height, width = size
        return np.zeros((height, width), dtype=np.uint8)


def is_point_in_mask(point, mask):
    """Check if a point falls within a binary mask"""
    x, y = int(point[0]), int(point[1])

    # Check bounds
    if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return False

    return mask[y, x] == 1


def is_point_in_bbox(point, bbox):
    """
    判断点是否在边界框内 (用于GUI任务)

    Args:
        point: [x, y] 点坐标
        bbox: [x1, y1, x2, y2] 边界框坐标 (xyxy格式)

    Returns:
        bool: 点是否在边界框内
    """
    if len(point) != 2 or len(bbox) != 4:
        return False

    x, y = point
    x1, y1, x2, y2 = bbox

    # 确保bbox坐标顺序正确
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    return x1 <= x <= x2 and y1 <= y <= y2


def get_box_center(box):
    """
    计算边界框的中心点 (用于GUI任务)

    Args:
        box: [x1, y1, x2, y2] 边界框 (xyxy格式)

    Returns:
        List[float]: [center_x, center_y] 中心点坐标
    """
    if len(box) != 4:
        return []

    x1, y1, x2, y2 = box

    # 确保坐标顺序正确
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return [center_x, center_y]


def extract_gui_predictions(extracted_predictions):
    """
    从extracted_predictions字典中提取所有点坐标和box坐标 (用于GUI任务)

    Args:
        extracted_predictions: 形如 {"street sign": [[209.45945945945945, 269.2692692692693]]} 或
                              {"button": [[x1, y1, x2, y2]]}

    Returns:
        Tuple[List[List[float]], List[List[float]]]: (点坐标列表, box坐标列表)
    """
    points = []
    boxes = []

    if not isinstance(extracted_predictions, dict):
        return points, boxes

    for key, value in extracted_predictions.items():
        if isinstance(value, list):
            for coord in value:
                if isinstance(coord, list):
                    try:
                        if len(coord) == 2:
                            # 点坐标格式 [x, y]
                            x, y = float(coord[0]), float(coord[1])
                            points.append([x, y])
                        elif len(coord) == 4:
                            # box坐标格式 [x1, y1, x2, y2]
                            x1, y1, x2, y2 = (
                                float(coord[0]),
                                float(coord[1]),
                                float(coord[2]),
                                float(coord[3]),
                            )
                            boxes.append([x1, y1, x2, y2])
                    except (ValueError, TypeError):
                        continue

    return points, boxes


def has_wrong_rejection(pred_boxes_dict):
    """
    检查预测中是否包含错误的拒绝（预测了None但不是hallucination任务）

    Args:
        pred_boxes_dict: 预测结果字典，形如 {"category": [boxes_or_None]}

    Returns:
        bool: 是否包含错误拒绝
    """
    for category, predictions in pred_boxes_dict.items():
        if not predictions:  # 空列表
            continue
        for pred in predictions:
            if pred is None or (isinstance(pred, str) and pred.lower() == "none"):
                return True
    return False


def calculate_gui_metrics(gt_box, extracted_predictions):
    """
    计算GUI定位任务的准确率

    Args:
        gt_box: Ground truth边界框 [x1, y1, x2, y2]
        extracted_predictions: 预测结果字典

    Returns:
        bool: 是否预测正确
    """
    # 提取预测点坐标和box坐标
    predicted_points, predicted_boxes = extract_gui_predictions(extracted_predictions)

    # 如果没有预测点和box，认为预测失败
    if not predicted_points and not predicted_boxes:
        return False

    # 如果GT为空或格式不正确，认为预测失败
    if not gt_box or len(gt_box) != 4:
        return False

    # 检查任意一个预测点是否落在GT框内
    for point in predicted_points:
        if is_point_in_bbox(point, gt_box):
            return True

    # 检查任意一个预测box的中心点是否落在GT框内
    for box in predicted_boxes:
        center_point = get_box_center(box)
        if center_point and is_point_in_bbox(center_point, gt_box):
            return True

    return False


def calculate_pointing_metrics(gt_masks_dict, pred_points_dict):
    """Calculate metrics for pointing task"""
    # Count total GT masks and predictions
    total_gt_count = sum(len(masks) for masks in gt_masks_dict.values())
    total_pred_count = sum(len(points) for points in pred_points_dict.values())

    if total_gt_count == 0:
        if total_pred_count == 0:
            return 1.0, 1.0, total_gt_count, total_pred_count
        else:
            return 0.0, 0.0, total_gt_count, total_pred_count

    if total_pred_count == 0:
        return 0.0, 0.0, total_gt_count, total_pred_count

    # Flatten all masks and points for matching
    all_gt_masks = []
    all_pred_points = []
    all_gt_categories = []
    all_pred_categories = []

    for category in gt_masks_dict:
        for mask_info in gt_masks_dict[category]:
            all_gt_masks.append(mask_info)
            all_gt_categories.append(category)

    for category in pred_points_dict:
        for point in pred_points_dict[category]:
            all_pred_points.append(point)
            all_pred_categories.append(category)

    # Pre-decode all masks to avoid repeated decoding
    decoded_masks = []
    for gt_mask_info in all_gt_masks:
        mask = decode_rle(gt_mask_info["counts"], gt_mask_info["size"])
        decoded_masks.append(mask)

    matches = 0
    used_preds = set()

    # For each GT mask, find the best matching prediction
    for i, (decoded_mask, gt_category) in enumerate(
        zip(decoded_masks, all_gt_categories)
    ):
        best_match = -1
        best_score = 0

        for j, (pred_point, pred_category) in enumerate(
            zip(all_pred_points, all_pred_categories)
        ):
            if j in used_preds:
                continue

            # Only match if categories match
            if gt_category != pred_category:
                continue

            # Check if point is in mask (using pre-decoded mask)
            if is_point_in_mask(pred_point, decoded_mask):
                # For pointing task, we consider it a match if the point is in the mask
                # and it's the first match for this GT mask
                if best_match == -1:
                    best_match = j
                    best_score = 1.0

        if best_match != -1:
            matches += 1
            used_preds.add(best_match)

    recall = matches / total_gt_count if total_gt_count > 0 else 0.0
    precision = matches / total_pred_count if total_pred_count > 0 else 0.0

    return recall, precision, total_gt_count, total_pred_count


def calculate_area(box):
    """Calculate area of a bounding box or polygon"""
    if is_polygon_format(box):
        # For polygon, use shapely to calculate area
        try:
            coords = [(box[i], box[i + 1]) for i in range(0, len(box), 2)]
            polygon = Polygon(coords)
            return polygon.area if polygon.is_valid else 0.0
        except:
            return 0.0
    else:
        # For bounding box
        return (box[2] - box[0]) * (box[3] - box[1])


def get_size_category(area):
    """Categorize object by size according to COCO standards"""
    if area < 32 * 32:
        return "small"
    elif area < 96 * 96:
        return "medium"
    else:
        return "large"


def get_gt_count_range(gt_count):
    """Categorize by GT count ranges"""
    if gt_count == 0:
        return "0"
    elif gt_count <= 5:
        return "1-5"
    elif gt_count <= 10:
        return "6-10"
    elif gt_count <= 15:
        return "11-15"
    elif gt_count <= 20:
        return "16-20"
    else:
        return "20+"


def normalize_category_name(category, task_name):
    """Normalize category name based on task type"""
    if task_name in [
        "common_object_detection",
        "long_tailed_object_detection",
        "dense_object_detection",
    ]:
        # For these tasks: lowercase and keep only letters
        return "".join(c.lower() for c in category if c.isalpha())
    else:
        # For other tasks: remove underscores and spaces
        return category.replace("_", "").replace(" ", "")


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (box1_area + box2_area - intersection)


def calculate_polygon_iou(poly1, poly2):
    """Calculate IoU between two polygons"""
    try:
        # Convert polygon coordinates to shapely Polygon objects
        # poly1 and poly2 are lists of coordinates [x0, y0, x1, y1, x2, y2, ...]
        if len(poly1) < 6 or len(poly2) < 6:  # Need at least 3 points (6 coordinates)
            return 0.0

        # Reshape coordinates to pairs
        coords1 = [(poly1[i], poly1[i + 1]) for i in range(0, len(poly1), 2)]
        coords2 = [(poly2[i], poly2[i + 1]) for i in range(0, len(poly2), 2)]

        # Create shapely polygons
        polygon1 = Polygon(coords1)
        polygon2 = Polygon(coords2)

        # Check if polygons are valid
        if not polygon1.is_valid or not polygon2.is_valid:
            return 0.0

        # Calculate intersection and union
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area

        if union == 0:
            return 0.0

        return intersection / union

    except Exception as e:
        # If there's any error in polygon calculation, return 0
        return 0.0


def is_polygon_format(coords):
    """Check if coordinates are in polygon format (more than 4 values)"""
    return len(coords) > 4


def calculate_detection_metrics(
    gt_boxes_dict, pred_boxes_dict, iou_threshold=0.5, use_polygon_iou=False
):
    """Calculate recall and precision for object detection tasks"""
    # Count total GT boxes across all categories
    total_gt_count = sum(len(boxes) for boxes in gt_boxes_dict.values())
    # Count total predicted boxes, excluding None values
    total_pred_count = sum(
        len([box for box in boxes if box is not None and box != "None"])
        for boxes in pred_boxes_dict.values()
    )

    if len(gt_boxes_dict) == 0:
        if total_pred_count == 0:
            return 1.0, 1.0, total_gt_count, total_pred_count
        else:
            return 0.0, 0.0, total_gt_count, total_pred_count

    if total_pred_count == 0:
        return 0.0, 0.0, total_gt_count, total_pred_count

    # Flatten all boxes for matching
    all_gt_boxes = []
    all_pred_boxes = []

    for category in gt_boxes_dict:
        all_gt_boxes.extend(gt_boxes_dict[category])
        # Filter out None values from predictions
        pred_boxes_for_category = pred_boxes_dict.get(category, [])
        valid_pred_boxes = [
            box for box in pred_boxes_for_category if box is not None and box != "None"
        ]
        all_pred_boxes.extend(valid_pred_boxes)

    matches = 0
    used_preds = set()

    for gt_box in all_gt_boxes:
        best_iou = 0
        best_pred_idx = -1

        for i, pred_box in enumerate(all_pred_boxes):
            if i in used_preds:
                continue

            # Choose IoU calculation method based on format
            if use_polygon_iou:
                iou = calculate_polygon_iou(gt_box, pred_box)
            else:
                iou = calculate_iou(gt_box, pred_box)

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pred_idx = i

        if best_pred_idx != -1:
            matches += 1
            used_preds.add(best_pred_idx)

    recall = matches / total_gt_count if total_gt_count > 0 else 0.0
    precision = matches / total_pred_count if total_pred_count > 0 else 0.0

    return recall, precision, total_gt_count, total_pred_count


def calculate_visual_prompt_metrics(gt_boxes_dict, pred_boxes_dict, iou_threshold=0.5):
    """Calculate metrics for visual prompt detection task"""
    # Calculate basic detection metrics
    recall, precision, gt_count, pred_count = calculate_detection_metrics(
        gt_boxes_dict, pred_boxes_dict, iou_threshold
    )

    # Calculate MAE (Mean Absolute Error) for box count
    mae = abs(pred_count - gt_count)

    # Calculate duplicate prediction ratio
    all_pred_boxes = []
    for category in pred_boxes_dict:
        all_pred_boxes.extend(pred_boxes_dict[category])

    duplicate_count = 0
    total_predictions = len(all_pred_boxes)

    if total_predictions > 1:
        for i in range(len(all_pred_boxes)):
            for j in range(i + 1, len(all_pred_boxes)):
                # Choose IoU calculation method based on format
                if is_polygon_format(all_pred_boxes[i]) or is_polygon_format(
                    all_pred_boxes[j]
                ):
                    iou = calculate_polygon_iou(all_pred_boxes[i], all_pred_boxes[j])
                else:
                    iou = calculate_iou(all_pred_boxes[i], all_pred_boxes[j])

                if iou > 0.9:  # High IoU threshold for duplicates
                    duplicate_count += 1

    duplicate_ratio = (
        duplicate_count / total_predictions if total_predictions > 0 else 0.0
    )

    return {
        "recall": recall,
        "precision": precision,
        "mae": mae,
        "duplicate_ratio": duplicate_ratio,
        "gt_count": gt_count,
        "pred_count": pred_count,
    }


def calculate_size_metrics(
    gt_boxes_dict, pred_boxes_dict, iou_threshold=0.5, use_polygon_iou=False
):
    """Calculate recall and precision for different object sizes"""
    size_metrics = {
        "small": {"gt": [], "pred": [], "matches": 0},
        "medium": {"gt": [], "pred": [], "matches": 0},
        "large": {"gt": [], "pred": [], "matches": 0},
    }

    # Process each category separately
    for category in gt_boxes_dict:
        gt_boxes = gt_boxes_dict[category]
        pred_boxes = pred_boxes_dict.get(category, [])

        # Categorize ground truth boxes by size
        for gt_box in gt_boxes:
            area = calculate_area(gt_box)
            size = get_size_category(area)
            size_metrics[size]["gt"].append(gt_box)

        # Categorize prediction boxes by size
        for pred_box in pred_boxes:
            area = calculate_area(pred_box)
            size = get_size_category(area)
            size_metrics[size]["pred"].append(pred_box)

    # Calculate matches for each size category
    for size in size_metrics:
        gt_boxes = size_metrics[size]["gt"]
        pred_boxes = size_metrics[size]["pred"]

        if len(gt_boxes) == 0:
            if len(pred_boxes) == 0:
                size_metrics[size]["recall"] = 1.0
                size_metrics[size]["precision"] = 1.0
            else:
                size_metrics[size]["recall"] = 0.0
                size_metrics[size]["precision"] = 0.0
            continue

        if len(pred_boxes) == 0:
            size_metrics[size]["recall"] = 0.0
            size_metrics[size]["precision"] = 0.0
            continue

        matches = 0
        used_preds = set()

        for gt_box in gt_boxes:
            best_iou = 0
            best_pred_idx = -1

            for i, pred_box in enumerate(pred_boxes):
                if i in used_preds:
                    continue

                # Choose IoU calculation method based on format
                if use_polygon_iou:
                    iou = calculate_polygon_iou(gt_box, pred_box)
                else:
                    iou = calculate_iou(gt_box, pred_box)

                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred_idx = i

            if best_pred_idx != -1:
                matches += 1
                used_preds.add(best_pred_idx)

        size_metrics[size]["recall"] = (
            matches / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
        )
        size_metrics[size]["precision"] = (
            matches / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
        )

    return size_metrics


def calculate_gt_count_metrics(
    gt_boxes_dict, pred_boxes_dict, iou_threshold=0.5, use_polygon_iou=False
):
    """Calculate metrics for different GT count ranges"""
    total_gt_count = sum(len(boxes) for boxes in gt_boxes_dict.values())
    gt_count_range = get_gt_count_range(total_gt_count)

    recall, precision, gt_count, pred_count = calculate_detection_metrics(
        gt_boxes_dict, pred_boxes_dict, iou_threshold, use_polygon_iou
    )

    return gt_count_range, recall, precision, gt_count, pred_count


class UniversalMetricsCalculator:
    """Universal metrics calculator for multiple tasks and datasets"""

    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(list))
        self.visual_prompt_metrics = defaultdict(lambda: {"maes": []})
        self.instruction_following_metrics = defaultdict(lambda: {"ratios": []})
        self.keypoint_metrics = defaultdict(
            lambda: {"ap_scores": [], "avg_oks": [], "gt_counts": [], "pred_counts": []}
        )
        self.hallucination_metrics = defaultdict(
            lambda: {"accuracies": [], "pred_counts": []}
        )
        self.gui_metrics = defaultdict(
            lambda: {"accuracies": [], "correct_counts": [], "total_counts": []}
        )
        self.wrong_rejection_metrics = defaultdict(
            lambda: {"wrong_rejections": [], "total_samples": []}
        )

    def calculate_metrics_for_sample(self, sample, iou_threshold=0.5):
        """Calculate metrics for a single sample"""
        task_name = sample["task_name"]
        dataset_name = sample["dataset_name"]
        gt_boxes = sample["gt"]
        pred_boxes = sample["extracted_predictions"]

        # Special handling for GUI task - gt is a list, not a dict
        if task_name == "gui":
            # For GUI task, keep gt_boxes as is (it's a list [x1, y1, x2, y2])
            # Only lowercase the prediction keys
            pred_boxes = {k.lower(): v for k, v in pred_boxes.items()}
        else:
            # lower case all the gt for other tasks
            gt_boxes = {k.lower(): v for k, v in gt_boxes.items()}
            pred_boxes = {k.lower(): v for k, v in pred_boxes.items()}

        # Special handling for different task types
        if task_name == "referring_object_detection":
            # For referring_object_detection, extract all boxes regardless of category names
            all_gt_boxes = []
            all_pred_boxes = []

            for category in gt_boxes:
                all_gt_boxes.extend(gt_boxes[category])

            for category in pred_boxes:
                all_pred_boxes.extend(pred_boxes[category])

            # Calculate metrics directly with flattened boxes
            gt_count = len(all_gt_boxes)
            pred_count = len(all_pred_boxes)

            if gt_count == 0:
                if pred_count == 0:
                    recall, precision = 1.0, 1.0
                else:
                    recall, precision = 0.0, 0.0
            elif pred_count == 0:
                recall, precision = 0.0, 0.0
            else:
                # Calculate matches
                matches = 0
                used_preds = set()

                for gt_box in all_gt_boxes:
                    best_iou = 0
                    best_pred_idx = -1

                    for i, pred_box in enumerate(all_pred_boxes):
                        if i in used_preds:
                            continue
                        iou = calculate_iou(gt_box, pred_box)
                        if iou > best_iou and iou >= iou_threshold:
                            best_iou = iou
                            best_pred_idx = i

                    if best_pred_idx != -1:
                        matches += 1
                        used_preds.add(best_pred_idx)

                recall = matches / gt_count if gt_count > 0 else 0.0
                precision = matches / pred_count if pred_count > 0 else 0.0
        elif task_name in [
            "task4_detect_all_in_polygon",
            "task2_detect_all_in_polygon_and_recog",
        ]:
            # For polygon tasks, use polygon IoU calculation
            # Process category names based on task type
            processed_gt_boxes = {}
            processed_pred_boxes = {}

            for category in gt_boxes:
                processed_category = normalize_category_name(category, task_name)
                processed_gt_boxes[processed_category] = gt_boxes[category]

            for category in pred_boxes:
                processed_category = normalize_category_name(category, task_name)
                processed_pred_boxes[processed_category] = pred_boxes[category]

            # Calculate basic detection metrics for polygon tasks
            recall, precision, gt_count, pred_count = calculate_detection_metrics(
                processed_gt_boxes,
                processed_pred_boxes,
                iou_threshold,
                use_polygon_iou=True,
            )
        elif task_name == "pointing":
            # For pointing task, use pointing metrics
            recall, precision, gt_count, pred_count = calculate_pointing_metrics(
                gt_boxes, pred_boxes
            )
        elif task_name == "pointing_referring":
            # For pointing task, use pointing metrics
            for gt_cate_names, things in gt_boxes.items():
                break

            if len(pred_boxes) > 0:
                for pred_cate_names, things in pred_boxes.items():
                    break
                new_pred_boxes = {gt_cate_names: pred_boxes[pred_cate_names]}
            else:
                new_pred_boxes = pred_boxes

            recall, precision, gt_count, pred_count = calculate_pointing_metrics(
                gt_boxes, new_pred_boxes
            )
        elif task_name == "keypoint":
            # For keypoint detection task, use keypoint metrics
            keypoint_results = calculate_keypoint_metrics_for_sample(
                gt_boxes, pred_boxes
            )

            # Store keypoint-specific metrics
            key = f"{task_name}_{dataset_name}"
            self.keypoint_metrics[key]["ap_scores"].append(
                keypoint_results["ap_results"]
            )
            self.keypoint_metrics[key]["avg_oks"].append(keypoint_results["avg_oks"])
            self.keypoint_metrics[key]["gt_counts"].append(
                keypoint_results["total_gt_instances"]
            )
            self.keypoint_metrics[key]["pred_counts"].append(
                keypoint_results["total_pred_instances"]
            )

            # Use different OKS thresholds to simulate IoU-like behavior for keypoint tasks
            # Map IoU thresholds to OKS thresholds for keypoint evaluation
            iou_to_oks_mapping = {
                0.5: "AP@0.50",
                0.55: "AP@0.50",
                0.6: "AP@0.50",
                0.65: "AP@0.75",
                0.7: "AP@0.75",
                0.75: "AP@0.75",
                0.8: "AP@0.90",
                0.85: "AP@0.90",
                0.9: "AP@0.90",
                0.95: "AP@0.95",
            }

            # Get the appropriate AP based on current IoU threshold (passed as parameter)
            oks_key = iou_to_oks_mapping.get(iou_threshold, "AP@0.50")
            main_ap = keypoint_results["ap_results"].get(oks_key, 0.0)

            recall = main_ap  # Use AP as recall for keypoint detection
            precision = main_ap  # Use AP as precision for keypoint detection
            gt_count = keypoint_results["total_gt_instances"]
            pred_count = keypoint_results["total_pred_instances"]
        elif task_name == "hallucination":
            # For hallucination task, model should NOT predict any boxes
            # Count total predicted boxes across all categories
            total_pred_count = sum(len(boxes) for boxes in pred_boxes.values())

            # Correct if no predictions are made (model didn't hallucinate)
            is_correct = 1.0 if total_pred_count == 0 else 0.0

            # For hallucination task:
            # - recall = accuracy (proportion of samples where model correctly predicted nothing)
            # - precision = accuracy (same as recall for this binary task)
            # - gt_count = 0 (no ground truth boxes since we're testing non-existent objects)
            # - pred_count = total predictions made
            recall = is_correct
            precision = is_correct
            gt_count = 0
            pred_count = total_pred_count
        elif task_name == "gui":
            # For GUI task, calculate if predicted points/boxes fall within GT box
            # GT format should be a single box [x1, y1, x2, y2] in sample["gt"]
            gt_box = sample.get("gt", [])

            # Calculate GUI accuracy
            is_correct = calculate_gui_metrics(gt_box, pred_boxes)

            # For GUI task:
            # - recall = accuracy (whether the prediction is correct)
            # - precision = accuracy (same as recall for this binary task)
            # - gt_count = 1 (always have one GT box)
            # - pred_count = total number of predictions made
            recall = 1.0 if is_correct else 0.0
            precision = 1.0 if is_correct else 0.0
            gt_count = 1
            pred_count = sum(len(coords) for coords in pred_boxes.values())
        else:
            # Process category names based on task type
            processed_gt_boxes = {}
            processed_pred_boxes = {}

            for category in gt_boxes:
                processed_category = normalize_category_name(category, task_name)
                processed_gt_boxes[processed_category] = gt_boxes[category]

            for category in pred_boxes:
                processed_category = normalize_category_name(category, task_name)
                processed_pred_boxes[processed_category] = pred_boxes[category]

            # Calculate basic detection metrics for other tasks
            recall, precision, gt_count, pred_count = calculate_detection_metrics(
                processed_gt_boxes, processed_pred_boxes, iou_threshold
            )

        # Store basic metrics
        key = f"{task_name}_{dataset_name}"
        self.results[key]["recalls"].append(recall)
        self.results[key]["precisions"].append(precision)
        self.results[key]["gt_counts"].append(gt_count)
        self.results[key]["pred_counts"].append(pred_count)

        # Store hallucination-specific metrics
        if task_name == "hallucination":
            if key not in self.hallucination_metrics:
                self.hallucination_metrics[key] = {"accuracies": [], "pred_counts": []}
            self.hallucination_metrics[key]["accuracies"].append(
                recall
            )  # recall = accuracy for hallucination task
            self.hallucination_metrics[key]["pred_counts"].append(pred_count)

        # Store GUI-specific metrics
        if task_name == "gui":
            if key not in self.gui_metrics:
                self.gui_metrics[key] = {
                    "accuracies": [],
                    "correct_counts": [],
                    "total_counts": [],
                }
            self.gui_metrics[key]["accuracies"].append(
                recall
            )  # recall = accuracy for GUI task
            self.gui_metrics[key]["correct_counts"].append(1 if recall == 1.0 else 0)
            self.gui_metrics[key]["total_counts"].append(1)

        # Calculate wrong rejection metrics (for all tasks except hallucination)
        if task_name != "hallucination":
            wrong_rejection = has_wrong_rejection(pred_boxes)
            if key not in self.wrong_rejection_metrics:
                self.wrong_rejection_metrics[key] = {
                    "wrong_rejections": [],
                    "total_samples": [],
                }
            self.wrong_rejection_metrics[key]["wrong_rejections"].append(
                1 if wrong_rejection else 0
            )
            self.wrong_rejection_metrics[key]["total_samples"].append(1)

        # Calculate additional metrics for visual_prompt_detection
        if task_name == "visual_prompt_detection":
            mae = abs(pred_count - gt_count)
            if key not in self.visual_prompt_metrics:
                self.visual_prompt_metrics[key] = {"maes": []}
            self.visual_prompt_metrics[key]["maes"].append(mae)

        # Calculate InstructionFollowing metric
        # Count unique categories in GT and predictions
        if task_name == "gui":
            # For GUI task, there's no category concept, skip instruction following metric
            gt_categories = set()
            pred_categories = set(pred_boxes.keys())
        else:
            gt_categories = set(gt_boxes.keys())
            pred_categories = set(pred_boxes.keys())

        # Special processing for common_object_detection and long_tailed_object_detection
        if task_name in ["common_object_detection", "long_tailed_object_detection"]:
            # Normalize categories: lowercase and keep only letters
            gt_categories_normalized = {
                normalize_category_name(cat, task_name): cat for cat in gt_categories
            }
            pred_categories_normalized = {
                normalize_category_name(cat, task_name): cat for cat in pred_categories
            }

            # Find matching categories based on normalized names
            matching_categories = set()
            for gt_norm, gt_orig in gt_categories_normalized.items():
                if gt_norm in pred_categories_normalized:
                    matching_categories.add(gt_orig)
        else:
            # For other tasks, use exact matching
            matching_categories = gt_categories.intersection(pred_categories)

        # Calculate the ratio of matching category counts
        if len(gt_categories) == 0:
            if len(pred_categories) == 0:
                instruction_following_ratio = 1.0
            else:
                instruction_following_ratio = 0.0
        else:
            instruction_following_ratio = len(matching_categories) / len(gt_categories)

        # Store instruction following metric
        if key not in self.instruction_following_metrics:
            self.instruction_following_metrics[key] = {"ratios": []}
        self.instruction_following_metrics[key]["ratios"].append(
            instruction_following_ratio
        )

    def calculate_all_metrics(self, data, iou_thresholds=[0.5]):
        """Calculate metrics for all samples with multiple IoU thresholds"""
        all_results = {}

        # Separate pointing, keypoint, hallucination and GUI tasks from other tasks for efficiency
        pointing_samples = []
        keypoint_samples = []
        hallucination_samples = []
        gui_samples = []
        other_samples = []

        for sample in data:
            if "task_name" not in sample:
                sample["task_name"] = "referring_object_detection"
            if sample["task_name"] == "pointing":
                pointing_samples.append(sample)
            elif sample["task_name"] == "keypoint":
                keypoint_samples.append(sample)
            elif sample["task_name"] == "hallucination":
                hallucination_samples.append(sample)
            elif sample["task_name"] == "gui":
                gui_samples.append(sample)
            else:
                other_samples.append(sample)

        print(
            f"Found {len(pointing_samples)} pointing samples, {len(keypoint_samples)} keypoint samples, {len(hallucination_samples)} hallucination samples, {len(gui_samples)} GUI samples, and {len(other_samples)} other samples"
        )

        # Process pointing tasks only once (IoU not relevant)
        if pointing_samples:
            print("Processing pointing tasks (IoU not relevant)...")
            self.results = defaultdict(lambda: defaultdict(list))
            self.visual_prompt_metrics = defaultdict(lambda: {"maes": []})
            self.instruction_following_metrics = defaultdict(lambda: {"ratios": []})
            self.hallucination_metrics = defaultdict(
                lambda: {"accuracies": [], "pred_counts": []}
            )
            self.gui_metrics = defaultdict(
                lambda: {"accuracies": [], "correct_counts": [], "total_counts": []}
            )
            self.wrong_rejection_metrics = defaultdict(
                lambda: {"wrong_rejections": [], "total_samples": []}
            )

            for sample in tqdm(pointing_samples, desc="Pointing tasks"):
                try:
                    self.calculate_metrics_for_sample(sample, 0.5)  # Use default IoU
                except Exception as e:
                    print(f"Error calculating metrics for pointing sample: {e}")

            # Store pointing results for all IoU thresholds (they're the same)
            pointing_results = {
                "basic_metrics": dict(self.results),
                "visual_prompt_metrics": dict(self.visual_prompt_metrics),
                "instruction_following_metrics": dict(
                    self.instruction_following_metrics
                ),
                "wrong_rejection_metrics": dict(self.wrong_rejection_metrics),
            }

        # Process keypoint tasks with different OKS thresholds (mapped from IoU thresholds)
        keypoint_results_by_iou = {}
        if keypoint_samples:
            print("Processing keypoint tasks with different OKS thresholds...")
            for iou in iou_thresholds:
                print(f"  Processing keypoint tasks for IoU/OKS threshold: {iou}")
                self.results = defaultdict(lambda: defaultdict(list))
                self.visual_prompt_metrics = defaultdict(lambda: {"maes": []})
                self.instruction_following_metrics = defaultdict(lambda: {"ratios": []})
                self.hallucination_metrics = defaultdict(
                    lambda: {"accuracies": [], "pred_counts": []}
                )
                self.gui_metrics = defaultdict(
                    lambda: {"accuracies": [], "correct_counts": [], "total_counts": []}
                )
                self.wrong_rejection_metrics = defaultdict(
                    lambda: {"wrong_rejections": [], "total_samples": []}
                )
                self.keypoint_metrics = defaultdict(
                    lambda: {
                        "ap_scores": [],
                        "avg_oks": [],
                        "gt_counts": [],
                        "pred_counts": [],
                    }
                )

                for sample in tqdm(
                    keypoint_samples, desc=f"Keypoint tasks (OKS threshold {iou})"
                ):
                    try:
                        self.calculate_metrics_for_sample(
                            sample, iou
                        )  # Use current IoU/OKS threshold
                    except Exception as e:
                        print(f"Error calculating metrics for keypoint sample: {e}")

                # Store keypoint results for this specific IoU/OKS threshold
                keypoint_results_by_iou[iou] = {
                    "basic_metrics": dict(self.results),
                    "visual_prompt_metrics": dict(self.visual_prompt_metrics),
                    "instruction_following_metrics": dict(
                        self.instruction_following_metrics
                    ),
                    "wrong_rejection_metrics": dict(self.wrong_rejection_metrics),
                    "keypoint_metrics": dict(self.keypoint_metrics),
                }

        # Process hallucination tasks only once (IoU not relevant)
        if hallucination_samples:
            print("Processing hallucination tasks (IoU not relevant)...")
            self.results = defaultdict(lambda: defaultdict(list))
            self.visual_prompt_metrics = defaultdict(lambda: {"maes": []})
            self.instruction_following_metrics = defaultdict(lambda: {"ratios": []})
            self.hallucination_metrics = defaultdict(
                lambda: {"accuracies": [], "pred_counts": []}
            )
            self.gui_metrics = defaultdict(
                lambda: {"accuracies": [], "correct_counts": [], "total_counts": []}
            )
            self.wrong_rejection_metrics = defaultdict(
                lambda: {"wrong_rejections": [], "total_samples": []}
            )

            for sample in tqdm(hallucination_samples, desc="Hallucination tasks"):
                try:
                    self.calculate_metrics_for_sample(sample, 0.5)  # Use default IoU
                except Exception as e:
                    print(f"Error calculating metrics for hallucination sample: {e}")

            # Store hallucination results for all IoU thresholds (they're the same)
            hallucination_results = {
                "basic_metrics": dict(self.results),
                "visual_prompt_metrics": dict(self.visual_prompt_metrics),
                "instruction_following_metrics": dict(
                    self.instruction_following_metrics
                ),
                "hallucination_metrics": dict(self.hallucination_metrics),
                "wrong_rejection_metrics": dict(self.wrong_rejection_metrics),
            }

        # Process GUI tasks only once (IoU not relevant)
        if gui_samples:
            print("Processing GUI tasks (IoU not relevant)...")
            self.results = defaultdict(lambda: defaultdict(list))
            self.visual_prompt_metrics = defaultdict(lambda: {"maes": []})
            self.instruction_following_metrics = defaultdict(lambda: {"ratios": []})
            self.gui_metrics = defaultdict(
                lambda: {"accuracies": [], "correct_counts": [], "total_counts": []}
            )
            self.wrong_rejection_metrics = defaultdict(
                lambda: {"wrong_rejections": [], "total_samples": []}
            )

            for sample in tqdm(gui_samples, desc="GUI tasks"):
                try:
                    self.calculate_metrics_for_sample(sample, 0.5)  # Use default IoU
                except Exception as e:
                    print(f"Error calculating metrics for GUI sample: {e}")

            # Store GUI results for all IoU thresholds (they're the same)
            gui_results = {
                "basic_metrics": dict(self.results),
                "visual_prompt_metrics": dict(self.visual_prompt_metrics),
                "instruction_following_metrics": dict(
                    self.instruction_following_metrics
                ),
                "gui_metrics": dict(self.gui_metrics),
                "wrong_rejection_metrics": dict(self.wrong_rejection_metrics),
            }

        # Process other tasks with multiple IoU thresholds
        for iou in iou_thresholds:
            print(f"Calculating metrics with IoU threshold: {iou}")

            # Reset results for this IoU threshold
            self.results = defaultdict(lambda: defaultdict(list))
            self.visual_prompt_metrics = defaultdict(lambda: {"maes": []})
            self.instruction_following_metrics = defaultdict(lambda: {"ratios": []})
            self.hallucination_metrics = defaultdict(
                lambda: {"accuracies": [], "pred_counts": []}
            )
            self.gui_metrics = defaultdict(
                lambda: {"accuracies": [], "correct_counts": [], "total_counts": []}
            )
            self.wrong_rejection_metrics = defaultdict(
                lambda: {"wrong_rejections": [], "total_samples": []}
            )

            # Process non-pointing samples
            for sample in tqdm(other_samples, desc=f"IoU {iou}"):
                try:
                    self.calculate_metrics_for_sample(sample, iou)
                except Exception as e:
                    print(f"Error calculating metrics for sample: {e}")

            # Store results for this IoU threshold
            all_results[iou] = {
                "basic_metrics": dict(self.results),
                "visual_prompt_metrics": dict(self.visual_prompt_metrics),
                "instruction_following_metrics": dict(
                    self.instruction_following_metrics
                ),
                "wrong_rejection_metrics": dict(self.wrong_rejection_metrics),
            }

            # Add pointing and keypoint results to this IoU threshold if they exist
            if pointing_samples:
                for key, value in pointing_results["basic_metrics"].items():
                    if key not in all_results[iou]["basic_metrics"]:
                        all_results[iou]["basic_metrics"][key] = value
                for key, value in pointing_results["visual_prompt_metrics"].items():
                    if key not in all_results[iou]["visual_prompt_metrics"]:
                        all_results[iou]["visual_prompt_metrics"][key] = value
                for key, value in pointing_results[
                    "instruction_following_metrics"
                ].items():
                    if key not in all_results[iou]["instruction_following_metrics"]:
                        all_results[iou]["instruction_following_metrics"][key] = value
                for key, value in pointing_results["wrong_rejection_metrics"].items():
                    if key not in all_results[iou]["wrong_rejection_metrics"]:
                        all_results[iou]["wrong_rejection_metrics"][key] = value

            if keypoint_samples and iou in keypoint_results_by_iou:
                keypoint_results = keypoint_results_by_iou[iou]
                for key, value in keypoint_results["basic_metrics"].items():
                    if key not in all_results[iou]["basic_metrics"]:
                        all_results[iou]["basic_metrics"][key] = value
                for key, value in keypoint_results["visual_prompt_metrics"].items():
                    if key not in all_results[iou]["visual_prompt_metrics"]:
                        all_results[iou]["visual_prompt_metrics"][key] = value
                for key, value in keypoint_results[
                    "instruction_following_metrics"
                ].items():
                    if key not in all_results[iou]["instruction_following_metrics"]:
                        all_results[iou]["instruction_following_metrics"][key] = value
                for key, value in keypoint_results["keypoint_metrics"].items():
                    if "keypoint_metrics" not in all_results[iou]:
                        all_results[iou]["keypoint_metrics"] = {}
                    all_results[iou]["keypoint_metrics"][key] = value
                for key, value in keypoint_results["wrong_rejection_metrics"].items():
                    if key not in all_results[iou]["wrong_rejection_metrics"]:
                        all_results[iou]["wrong_rejection_metrics"][key] = value

            if hallucination_samples:
                for key, value in hallucination_results["basic_metrics"].items():
                    if key not in all_results[iou]["basic_metrics"]:
                        all_results[iou]["basic_metrics"][key] = value
                for key, value in hallucination_results[
                    "visual_prompt_metrics"
                ].items():
                    if key not in all_results[iou]["visual_prompt_metrics"]:
                        all_results[iou]["visual_prompt_metrics"][key] = value
                for key, value in hallucination_results[
                    "instruction_following_metrics"
                ].items():
                    if key not in all_results[iou]["instruction_following_metrics"]:
                        all_results[iou]["instruction_following_metrics"][key] = value
                for key, value in hallucination_results[
                    "hallucination_metrics"
                ].items():
                    if "hallucination_metrics" not in all_results[iou]:
                        all_results[iou]["hallucination_metrics"] = {}
                    all_results[iou]["hallucination_metrics"][key] = value
                for key, value in hallucination_results[
                    "wrong_rejection_metrics"
                ].items():
                    if key not in all_results[iou]["wrong_rejection_metrics"]:
                        all_results[iou]["wrong_rejection_metrics"][key] = value

            if gui_samples:
                for key, value in gui_results["basic_metrics"].items():
                    if key not in all_results[iou]["basic_metrics"]:
                        all_results[iou]["basic_metrics"][key] = value
                for key, value in gui_results["visual_prompt_metrics"].items():
                    if key not in all_results[iou]["visual_prompt_metrics"]:
                        all_results[iou]["visual_prompt_metrics"][key] = value
                for key, value in gui_results["instruction_following_metrics"].items():
                    if key not in all_results[iou]["instruction_following_metrics"]:
                        all_results[iou]["instruction_following_metrics"][key] = value
                for key, value in gui_results["gui_metrics"].items():
                    if "gui_metrics" not in all_results[iou]:
                        all_results[iou]["gui_metrics"] = {}
                    all_results[iou]["gui_metrics"][key] = value
                for key, value in gui_results["wrong_rejection_metrics"].items():
                    if key not in all_results[iou]["wrong_rejection_metrics"]:
                        all_results[iou]["wrong_rejection_metrics"][key] = value

        return all_results

    def print_results(self, all_results):
        """Print simplified results with COCO-style metrics"""
        print("\n" + "=" * 180)
        print("UNIVERSAL METRICS CALCULATION RESULTS".center(180))
        print("=" * 180)

        # Organize metrics by dataset and IoU threshold
        dataset_metrics_by_iou = {}

        for iou, results in all_results.items():
            for key, metrics in results["basic_metrics"].items():
                if metrics["recalls"]:
                    if key not in dataset_metrics_by_iou:
                        dataset_metrics_by_iou[key] = {}

                    avg_recall = mean(metrics["recalls"])
                    avg_precision = mean(metrics["precisions"])
                    f1_score = (
                        2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                        if (avg_precision + avg_recall) > 0
                        else 0.0
                    )

                    dataset_metrics_by_iou[key][iou] = {
                        "precision": avg_precision,
                        "recall": avg_recall,
                        "f1": f1_score,
                        "samples": len(metrics["recalls"]),
                    }

        # Calculate MAE for visual prompt tasks (aggregated across all IoU thresholds)
        vp_mae_results = {}
        for iou, results in all_results.items():
            for key, vp_metrics in results["visual_prompt_metrics"].items():
                if vp_metrics["maes"]:
                    if key not in vp_mae_results:
                        vp_mae_results[key] = []
                    vp_mae_results[key].extend(vp_metrics["maes"])

        # Calculate InstructionFollowing metrics (aggregated across all IoU thresholds)
        instruction_following_results = {}
        for iou, results in all_results.items():
            for key, if_metrics in results["instruction_following_metrics"].items():
                if if_metrics["ratios"]:
                    if key not in instruction_following_results:
                        instruction_following_results[key] = []
                    instruction_following_results[key].extend(if_metrics["ratios"])

        # Calculate Keypoint metrics (aggregated across all IoU thresholds)
        keypoint_results = {}
        for iou, results in all_results.items():
            if "keypoint_metrics" in results:
                for key, kp_metrics in results["keypoint_metrics"].items():
                    if kp_metrics["ap_scores"]:
                        if key not in keypoint_results:
                            keypoint_results[key] = {"ap_scores": [], "avg_oks": []}
                        keypoint_results[key]["ap_scores"].extend(
                            kp_metrics["ap_scores"]
                        )
                        keypoint_results[key]["avg_oks"].extend(kp_metrics["avg_oks"])

        # Calculate Hallucination metrics (aggregated across all IoU thresholds)
        hallucination_results = {}
        for iou, results in all_results.items():
            if "hallucination_metrics" in results:
                for key, hall_metrics in results["hallucination_metrics"].items():
                    if hall_metrics["accuracies"]:
                        if key not in hallucination_results:
                            hallucination_results[key] = {
                                "accuracies": [],
                                "pred_counts": [],
                            }
                        hallucination_results[key]["accuracies"].extend(
                            hall_metrics["accuracies"]
                        )
                        hallucination_results[key]["pred_counts"].extend(
                            hall_metrics["pred_counts"]
                        )

        # Calculate GUI metrics (aggregated across all IoU thresholds)
        gui_results = {}
        for iou, results in all_results.items():
            if "gui_metrics" in results:
                for key, gui_metrics in results["gui_metrics"].items():
                    if gui_metrics["accuracies"]:
                        if key not in gui_results:
                            gui_results[key] = {
                                "accuracies": [],
                                "correct_counts": [],
                                "total_counts": [],
                            }
                        gui_results[key]["accuracies"].extend(gui_metrics["accuracies"])
                        gui_results[key]["correct_counts"].extend(
                            gui_metrics["correct_counts"]
                        )
                        gui_results[key]["total_counts"].extend(
                            gui_metrics["total_counts"]
                        )

        # Calculate Wrong Rejection metrics (aggregated across all IoU thresholds)
        wrong_rejection_results = {}
        for iou, results in all_results.items():
            if "wrong_rejection_metrics" in results:
                for key, wr_metrics in results["wrong_rejection_metrics"].items():
                    if wr_metrics["wrong_rejections"]:
                        if key not in wrong_rejection_results:
                            wrong_rejection_results[key] = {
                                "wrong_rejections": [],
                                "total_samples": [],
                            }
                        wrong_rejection_results[key]["wrong_rejections"].extend(
                            wr_metrics["wrong_rejections"]
                        )
                        wrong_rejection_results[key]["total_samples"].extend(
                            wr_metrics["total_samples"]
                        )

        # Find the maximum length of task_dataset names for proper alignment
        max_name_length = (
            max(len(key) for key in dataset_metrics_by_iou.keys())
            if dataset_metrics_by_iou
            else 30
        )
        name_width = min(max_name_length + 5, 60)  # Cap at 60 characters

        # Print header with IoU thresholds
        header = f"{'Task_Dataset':<{name_width}} | {'IoU=0.5':<20} | {'IoU=0.9':<20} | {'mIoU':<20} | {'MAE':<12} | {'InstFollow':<12} | {'Keypoint':<15} | {'Halluc_Acc':<12} | {'GUI_Acc':<10} | {'WrongRej':<10}"
        print(header)
        print("-" * len(header))

        for key, iou_metrics in dataset_metrics_by_iou.items():
            # Check if this is a pointing task
            is_pointing_task = key.startswith("pointing_")

            if is_pointing_task:
                # For pointing tasks, use any available IoU threshold (they should all be the same)
                any_iou = list(iou_metrics.keys())[0] if iou_metrics else 0.5
                pointing_metrics = iou_metrics.get(
                    any_iou, {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                )

                # For pointing tasks, we only show the main metrics (no IoU variations)
                pointing_str = f"P:{pointing_metrics['precision']:.3f} R:{pointing_metrics['recall']:.3f} F1:{pointing_metrics['f1']:.3f}"

                # Get Wrong Rejection metrics for pointing tasks
                wrong_rej_str = "-"
                if key in wrong_rejection_results:
                    wrong_rejections = wrong_rejection_results[key]["wrong_rejections"]
                    total_samples = wrong_rejection_results[key]["total_samples"]
                    if total_samples:
                        wrong_rej_rate = sum(wrong_rejections) / sum(total_samples)
                        wrong_rej_str = f"{wrong_rej_rate:.4f}"

                # Format the line for pointing tasks
                line = f"{key:<{name_width}} | {pointing_str:<20} | {'N/A':<20} | {'N/A':<20} | {'N/A':<12} | {'N/A':<12} | {'N/A':<15} | {'N/A':<12} | {'N/A':<10} | {wrong_rej_str:<10}"
                print(line)
            else:
                # Get metrics for IoU=0.5
                iou_05_metrics = iou_metrics.get(
                    0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                )
                iou_05_str = f"P:{iou_05_metrics['precision']:.3f} R:{iou_05_metrics['recall']:.3f} F1:{iou_05_metrics['f1']:.3f}"

                # Get metrics for IoU=0.95
                iou_095_metrics = iou_metrics.get(
                    0.95, {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                )
                iou_095_str = f"P:{iou_095_metrics['precision']:.3f} R:{iou_095_metrics['recall']:.3f} F1:{iou_095_metrics['f1']:.3f}"

                # Calculate mIoU (mean across all IoU thresholds)
                all_precisions = [
                    metrics["precision"] for metrics in iou_metrics.values()
                ]
                all_recalls = [metrics["recall"] for metrics in iou_metrics.values()]
                all_f1s = [metrics["f1"] for metrics in iou_metrics.values()]

                mprecision = mean(all_precisions) if all_precisions else 0.0
                mrecall = mean(all_recalls) if all_recalls else 0.0
                mf1 = mean(all_f1s) if all_f1s else 0.0
                miou_str = f"P:{mprecision:.3f} R:{mrecall:.3f} F1:{mf1:.3f}"

                # Get MAE for visual prompt tasks
                mae_str = "-"
                if key in vp_mae_results:
                    avg_mae = mean(vp_mae_results[key])
                    mae_str = f"{avg_mae:.4f}"

                # Get InstructionFollowing metric
                inst_follow_str = "-"
                if key in instruction_following_results:
                    avg_inst_follow = mean(instruction_following_results[key])
                    inst_follow_str = f"{avg_inst_follow:.4f}"

                # Get Keypoint metrics
                keypoint_str = "-"
                if key in keypoint_results:
                    # Calculate average AP@0.5 and average OKS
                    ap_scores = keypoint_results[key]["ap_scores"]
                    avg_oks = keypoint_results[key]["avg_oks"]

                    if ap_scores:
                        # Extract AP@0.5 from each sample
                        ap_05_scores = []
                        for ap_dict in ap_scores:
                            if "AP@0.50" in ap_dict:
                                ap_05_scores.append(ap_dict["AP@0.50"])

                        avg_ap_05 = mean(ap_05_scores) if ap_05_scores else 0.0
                        avg_oks_score = mean(avg_oks) if avg_oks else 0.0
                        keypoint_str = f"AP:{avg_ap_05:.3f} OKS:{avg_oks_score:.3f}"

                # Get Hallucination metrics
                hallucination_str = "-"
                if key in hallucination_results:
                    # Calculate average accuracy and average prediction count
                    accuracies = hallucination_results[key]["accuracies"]
                    pred_counts = hallucination_results[key]["pred_counts"]

                    if accuracies:
                        avg_accuracy = mean(accuracies)
                        avg_pred_count = mean(pred_counts) if pred_counts else 0.0
                        hallucination_str = f"{avg_accuracy:.4f}"

                # Get GUI metrics
                gui_str = "-"
                if key in gui_results:
                    # Calculate average accuracy
                    accuracies = gui_results[key]["accuracies"]
                    correct_counts = gui_results[key]["correct_counts"]
                    total_counts = gui_results[key]["total_counts"]

                    if accuracies:
                        avg_accuracy = mean(accuracies)
                        total_correct = sum(correct_counts)
                        total_samples = sum(total_counts)
                        gui_str = f"{avg_accuracy:.4f}"

                # Get Wrong Rejection metrics
                wrong_rej_str = "-"
                if key in wrong_rejection_results:
                    wrong_rejections = wrong_rejection_results[key]["wrong_rejections"]
                    total_samples = wrong_rejection_results[key]["total_samples"]
                    if total_samples:
                        wrong_rej_rate = sum(wrong_rejections) / sum(total_samples)
                        wrong_rej_str = f"{wrong_rej_rate:.4f}"

                # Format the line with proper alignment
                line = f"{key:<{name_width}} | {iou_05_str:<20} | {iou_095_str:<20} | {miou_str:<20} | {mae_str:<12} | {inst_follow_str:<12} | {keypoint_str:<15} | {hallucination_str:<12} | {gui_str:<10} | {wrong_rej_str:<10}"
                print(line)

        print("=" * len(header))


def get_args():
    parser = argparse.ArgumentParser(
        description="Universal metrics calculator for multiple tasks"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/point_eval/RefCOCOg_test/answer.jsonl",
        help="Path to prediction JSONL file",
    )
    parser.add_argument(
        "--iou_thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        help="IoU thresholds to evaluate (default: [0.5, 0.75, 0.9] for faster evaluation)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/point_eval/RefCOCOg_test/eval_results.json",
        help="Path to save detailed results JSON (optional)",
    )
    parser.add_argument(
        "--auto_detect_pointing",
        action="store_true",
        default=True,
        help="Automatically detect pointing tasks and use single IoU threshold",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Load data
    print(f"Loading data from: {args.data_path}")
    with open(args.data_path, "r") as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} samples")

    # Check if auto-detect pointing is enabled
    if args.auto_detect_pointing:
        # Check if any sample is a pointing task
        pointing_tasks = [
            sample
            for sample in data
            if sample.get("task_name") in ["pointing", "pointing_referring"]
        ]
        if pointing_tasks:
            print(
                f"Detected {len(pointing_tasks)} pointing tasks. Using single IoU threshold [0.5] for efficiency."
            )
            iou_thresholds = [0.5]
        else:
            print("No pointing tasks detected. Using full IoU threshold range.")
            iou_thresholds = args.iou_thresholds
    else:
        # Always use the full IoU threshold range when auto-detect is disabled
        # The calculate_all_metrics method will handle mixed tasks properly
        iou_thresholds = args.iou_thresholds
        pointing_tasks = [
            sample
            for sample in data
            if sample.get("task_name") in ["pointing", "pointing_referring"]
        ]
        if pointing_tasks:
            print(
                f"Found {len(pointing_tasks)} pointing tasks mixed with other tasks. "
                f"Will calculate metrics for all IoU thresholds: {iou_thresholds}"
            )

    # Initialize calculator
    calculator = UniversalMetricsCalculator()

    # Calculate metrics
    all_results = calculator.calculate_all_metrics(data, iou_thresholds)

    # Print results
    calculator.print_results(all_results)

    # Save detailed results if requested
    if args.output_path:
        print(f"\nSaving detailed results to: {args.output_path}")
        if not os.path.exists(os.path.dirname(args.output_path)):
            os.makedirs(os.path.dirname(args.output_path))
        with open(args.output_path, "w") as f:
            json.dump(
                all_results,
                f,
                indent=2,
                default=lambda x: float(x) if isinstance(x, (int, float)) else str(x),
            )


def calculate_keypoint_distance(pred_point, gt_point):
    """Calculate Euclidean distance between two keypoints"""
    if pred_point is None or gt_point is None:
        return float("inf")
    return np.sqrt(
        (pred_point[0] - gt_point[0]) ** 2 + (pred_point[1] - gt_point[1]) ** 2
    )


def calculate_oks(gt_bbox, gt_keypoints, pred_bbox, pred_keypoints, sigma=0.025):
    """
    Calculate Object Keypoint Similarity (OKS)

    Args:
        gt_bbox: Ground truth bounding box [x1, y1, x2, y2]
        gt_keypoints: Ground truth keypoints dict
        pred_bbox: Predicted bounding box [x1, y1, x2, y2]
        pred_keypoints: Predicted keypoints dict
        sigma: Standard deviation for OKS calculation

    Returns:
        OKS score
    """
    if not gt_keypoints or not pred_keypoints:
        return 0.0

    # Calculate bbox area for normalization
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    if gt_area <= 0:
        return 0.0

    # Only consider keypoints that exist in GT (don't penalize extra predictions)
    gt_keypoint_names = set(gt_keypoints.keys())

    total_weight = 0
    total_score = 0

    for kp_name in gt_keypoint_names:
        gt_kp = gt_keypoints.get(kp_name)
        pred_kp = pred_keypoints.get(kp_name)

        # Skip if GT keypoint is missing (shouldn't happen since we iterate over GT keys)
        # or if predicted keypoint is missing (model didn't predict this GT keypoint)
        if gt_kp is None or pred_kp is None:
            continue

        # Calculate distance
        distance = calculate_keypoint_distance(pred_kp, gt_kp)

        # Calculate OKS for this keypoint
        # Using a simplified approach where all keypoints have equal weight
        weight = 1.0
        kp_score = weight * np.exp(-(distance**2) / (2 * sigma**2 * gt_area))

        total_score += kp_score
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return total_score / total_weight


def calculate_keypoint_ap(
    gt_instances, pred_instances, oks_thresholds=[0.5, 0.75, 0.9, 0.95]
):
    """
    Calculate Average Precision for keypoint detection

    Args:
        gt_instances: List of ground truth instances with bbox and keypoints
        pred_instances: List of predicted instances with bbox and keypoints
        oks_thresholds: List of OKS thresholds for AP calculation

    Returns:
        Dictionary with AP scores for each threshold
    """
    if not gt_instances:
        # If no GT instances with keypoints, return perfect score regardless of predictions
        # This means we don't penalize the model for predicting keypoints when GT has none
        return {f"AP@{thresh:.2f}": 1.0 for thresh in oks_thresholds}

    if not pred_instances:
        return {f"AP@{thresh:.2f}": 0.0 for thresh in oks_thresholds}

    ap_results = {}

    for threshold in oks_thresholds:
        # Calculate OKS for all GT-prediction pairs
        all_scores = []
        all_matches = []

        for gt_idx, gt_instance in enumerate(gt_instances):
            gt_bbox = gt_instance.get("bbox", [0, 0, 1, 1])
            gt_keypoints = gt_instance.get("keypoints", {})

            best_score = 0.0
            best_pred_idx = -1

            for pred_idx, pred_instance in enumerate(pred_instances):
                pred_bbox = pred_instance.get("bbox", [0, 0, 1, 1])
                pred_keypoints = pred_instance.get("keypoints", {})

                oks_score = calculate_oks(
                    gt_bbox, gt_keypoints, pred_bbox, pred_keypoints
                )

                if oks_score > best_score:
                    best_score = oks_score
                    best_pred_idx = pred_idx

            all_scores.append(best_score)
            all_matches.append(best_score >= threshold)

        # Calculate precision and recall
        if len(all_matches) == 0:
            ap = 0.0
        else:
            # Simple AP calculation: fraction of GT instances that have a match
            ap = sum(all_matches) / len(all_matches)

        ap_results[f"AP@{threshold:.2f}"] = ap

    return ap_results


def calculate_keypoint_metrics_for_sample(gt_data, pred_data):
    """
    Calculate keypoint detection metrics for a single sample

    Args:
        gt_data: Ground truth data in the format {"hand": [{"bbox": [...], "keypoints": {...}, "phrase": "..."}]}
        pred_data: Prediction data in the same format

    Returns:
        Dictionary with keypoint metrics
    """
    # Extract instances from both GT and predictions
    gt_instances = []
    pred_instances = []

    # Process GT data - only include instances with keypoints
    for category, instances in gt_data.items():
        for instance in instances:
            # Skip GT instances that have no keypoints (empty dict or None)
            keypoints = instance.get("keypoints", {})
            if keypoints and len(keypoints) > 0:
                gt_instances.append(instance)

    # Process prediction data
    for category, instances in pred_data.items():
        for instance in instances:
            pred_instances.append(instance)

    # Calculate AP at different thresholds
    ap_results = calculate_keypoint_ap(gt_instances, pred_instances)

    # Calculate additional metrics
    total_gt_instances = len(gt_instances)
    total_pred_instances = len(pred_instances)

    # Calculate average OKS for matched pairs
    avg_oks = 0.0
    matched_pairs = 0

    for gt_instance in gt_instances:
        gt_bbox = gt_instance.get("bbox", [0, 0, 1, 1])
        gt_keypoints = gt_instance.get("keypoints", {})

        best_oks = 0.0
        for pred_instance in pred_instances:
            pred_bbox = pred_instance.get("bbox", [0, 0, 1, 1])
            pred_keypoints = pred_instance.get("keypoints", {})

            oks_score = calculate_oks(gt_bbox, gt_keypoints, pred_bbox, pred_keypoints)
            best_oks = max(best_oks, oks_score)

        if best_oks > 0.5:  # Consider it a match if OKS > 0.5
            avg_oks += best_oks
            matched_pairs += 1

    avg_oks = avg_oks / matched_pairs if matched_pairs > 0 else 0.0

    return {
        "ap_results": ap_results,
        "total_gt_instances": total_gt_instances,
        "total_pred_instances": total_pred_instances,
        "avg_oks": avg_oks,
        "matched_pairs": matched_pairs,
    }


if __name__ == "__main__":
    main()

#!/bin/bash

# COCO Evaluation Pipeline - Simple Version
# This script runs the complete COCO evaluation pipeline with fixed parameters

set -e  # Exit on any error

# Fixed parameters
MODEL_PATH="IDEA-Research/Rex-Omni"
TEST_JSONL="Mountchicken/Rex-Omni-Eval/annotations/box_eval/COCO.jsonl"
IMAGE_ROOT="Mountchicken/Rex-Omni-Eval"
COCO_JSON="Mountchicken/Rex-Omni-Eval/coco/instances_val2017.json"
OUTPUT_DIR="Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/box_eval/COCO"
START_IDX=0
END_IDX=-1
MAX_TOKENS=4096
MIN_PIXELS=$((16 * 28 * 28))
MAX_PIXELS=$((2560 * 28 * 28))

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set file paths
PRED_JSONL="$OUTPUT_DIR/answer.jsonl"
FASTEVAL_TSV="$OUTPUT_DIR/fast_eval.tsv"

echo "🚀 Starting COCO Evaluation Pipeline"
echo "======================================"
echo "Model: $MODEL_PATH"
echo "Test data: $TEST_JSONL"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Step 1: Run inference
echo "📝 Step 1: Running inference with Rex-Omni..."
python evaluation/inference_text_prompt.py \
    --model_path "$MODEL_PATH" \
    --test_jsonl_path "$TEST_JSONL" \
    --image_root_dir "$IMAGE_ROOT" \
    --save_path "$PRED_JSONL" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --max_new_tokens "$MAX_TOKENS" \
    --min_pixels "$MIN_PIXELS" \
    --max_pixels "$MAX_PIXELS" \
    --question "Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format."

echo "✅ Inference completed successfully!"
echo ""

# Step 2: Convert to FastEval TSV format
echo "🔄 Step 2: Converting predictions to FastEval TSV format..."
python evaluation/utils/convert_coco_lvis_to_standard_format.py \
    --our_pred_jsonl "$PRED_JSONL" \
    --coco_json "$COCO_JSON" \
    --out_tsv "$FASTEVAL_TSV" \
    --positive_only

echo "✅ Format conversion completed successfully!"
echo ""

# Step 3: Evaluate using FastEval
echo "📊 Step 3: Running FastEval evaluation..."
python evaluation/metrics/coco_lvis_metric.py \
    --gt "$COCO_JSON" \
    --pred_tsv "$FASTEVAL_TSV"

echo "✅ Evaluation completed successfully!"
echo ""

echo "🎉 COCO Evaluation Pipeline completed!"
echo "======================================"
echo "Results saved to:"
echo "  - Predictions: $PRED_JSONL"
echo "  - FastEval TSV: $FASTEVAL_TSV"

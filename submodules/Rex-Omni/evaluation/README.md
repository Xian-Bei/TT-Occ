# Rex-Omni Evaluation Guide

This guide shows how to download evaluation data, unpack images, and run Rex-Omni evaluations across datasets and task types.

### 1 Install FastEvaluate (required for COCO/LVIS metrics)

```bash
cd evaluation/fastevaluate
pip install -e .
```

### 2 Download datasets

- Source: `https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData`
- After downloading, the directory layout should look like `Rex-Omni-Eval/` with images packaged as `.tar.gz` files. Example on disk:

```
/.../Rex-Omni-Eval
  *.tar.gz               # per-dataset image archives (e.g., coco.tar.gz, hiertext.tar.gz, ...)
  _annotations/          # JSONL annotations (multiple eval types)
  _rex_omni_eval_results # The evaluation results of Rex-Omni
```

Unpack the image archives before running:

```bash
cd /path/to/Rex-Omni-Eval
for f in *.tar.gz; do
  echo "Extracting $f" && tar -xzf "$f"
done
```

### 3 Evaluation
The evaluation is seperated into two categories:
1. COCO/LVIS text-prompt evaluation
2. Other datasets (box/point/visual-prompt)

#### COCO/LVIS text-prompt evaluation in box format
For text prompt evaluation on COCO and LVIS dataset (box format), run the following script

- For COCO evaluation
  
```
bash evaluation/scrpts/eval_coco.sh \
    --model_path IDEA-Research/Rex-Omni \
    --test_jsonl Mountchicken/Rex-Omni-Eval/annotations/box_eval/COCO.jsonl \
    --image_root Mountchicken/Rex-Omni-Eval \
    --coco_json Mountchicken/Rex-Omni-Eval/coco/instances_val2017.json \
    --output_dir Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/text_prompt_eval/COCO \
```

- For LVIS evaluation

```
bash evaluation/scrpts/eval_lvis.sh \
    --model_path IDEA-Research/Rex-Omni \
    --test_jsonl Mountchicken/Rex-Omni-Eval/annotations/box_eval/LVIS.jsonl \
    --image_root Mountchicken/Rex-Omni-Eval \
    --lvis_json Mountchicken/Rex-Omni-Eval/coco/lvis_v1_val_with_filename2.json \
    --output_dir Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/text_prompt_eval/COCO \

```

#### Other datasets and task (box/point/visual-prompt)

- For text prompt task (output box)

```
bash evaluation/scrpts/eval_others.sh \
    --dataset Dense200 \ # choice in Dense200, DocLayNet, HierText, HumanRef, IC15, M6Doc, RefCOCOg_test, RefCOCOg_val, SROIE, TotalText, VisDrone
    --eval_type box_eval \
    --model_path IDEA-Research/Rex-Omni \
    --image_root Mountchicken/Rex-Omni-Eval \
    --output_base Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/box_eval/
```

- For text prompt task (output point)
```
bash evaluation/scrpts/eval_others.sh \
    --dataset COCO \ # choice in COCO, Dense200, HumanRef, LVIS, RefCOCOg_test, RefCOCOg_val, VisDrone
    --eval_type point_eval \
    --model_path IDEA-Research/Rex-Omni \
    --image_root Mountchicken/Rex-Omni-Eval \
    --output_base Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/point_eval/
```


- For visual prompt task
```
bash evaluation/scrpts/eval_others.sh \
    --dataset COCO \ # choice in COCO, Dense200, FSCD_test, LVIS VisDrone
    --eval_type visual_prompt_eval \
    --model_path IDEA-Research/Rex-Omni \
    --image_root Mountchicken/Rex-Omni-Eval \
    --output_base Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/visual_prompt_eval/
```



import json

gt_path = "Mountchicken/Rex-Omni-Eval/_annotations/point_eval/HumanRef.jsonl"

image_path2mask = {}
with open(gt_path, "r") as f:
    for line in f:
        data = json.loads(line)
        image_path = data["image_path"]
        mask_path = data["gt_mask"]
        categories = data["categories"]
        categories = "".join(categories)
        image_path2mask[f"{image_path}_{categories}"] = mask_path


pred_path = (
    "Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/point_eval/HumanRef/answer.jsonl"
)

new_annos = []
with open(pred_path, "r") as f:
    for line in f:
        data = json.loads(line)
        image_path = data["image_path"]
        prompt = data["question"]
        categories = prompt.replace("point to ", "")[:-1]

        data["gt"] = image_path2mask[f"{image_path}_{categories}"]
        new_annos.append(data)

with open(pred_path, "w") as f:
    for data in new_annos:
        f.write(json.dumps(data) + "\n")


# import json
# import os

# from tqdm import tqdm

# anno_path = (
#     "Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/point_eval/Dense200/answer.jsonl"
# )
# # task_name = "referring_object_detection"
# task_name = "pointing"
# # task_name = "hallucination"

# new_data = []
# rejection_data = []
# with open(anno_path, "r") as f:
#     for line in tqdm(f):
#         data = json.loads(line)
#         data["task_name"] = task_name
#         new_data.append(data)

# with open(anno_path, "w") as f:
#     for data in new_data:
#         f.write(json.dumps(data) + "\n")

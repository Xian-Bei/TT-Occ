import json

anno_path = "Mountchicken/Rex-Omni-Eval/_rex_omni_eval_results/visual_prompt_eval/LVIS/answer.jsonl"

new_annos = []
existe_path = []

with open(anno_path, "r") as f:
    for line in f:
        data = json.loads(line)
        image_path = data["image_path"]
        if image_path in existe_path:
            continue
        existe_path.append(image_path)
        new_annos.append(data)

with open(anno_path, "w") as f:
    for data in new_annos:
        f.write(json.dumps(data) + "\n")

import os
import json
from tqdm import tqdm

LABEL_DIR = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/adata_coco_temprary/label"
OUTPUT_PATH = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/label.json"

merged_labels = {}
next_id = 1

json_files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith(".json")])

for file_name in tqdm(json_files, desc="Merging label JSONs"):
    file_path = os.path.join(LABEL_DIR, file_name)
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        for label in data.keys():
            if label not in merged_labels:
                merged_labels[label] = next_id
                next_id += 1

    except Exception as e:
        print(f"[WARN] Error read file {file_name}: {e}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(merged_labels, f, indent=4, ensure_ascii=False)

print(f"[SUCCESS] Merged {len(json_files)} files into {OUTPUT_PATH}")
print(f"[INFO] Total label: {len(merged_labels)}")
print("[DONE] Labels:", merged_labels)

import os
import json
import difflib
from tqdm import tqdm

RAW_DIR = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/raw"
ANNOTATED_DIR = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/annotated"
COCO_JSON_DIR = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/annotated/coco_json/coco3"
OUTPUT_PATH = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/coco_dataset/coco_dataset3.json"

start_img_id = 0   # ID anh
start_ann_id = 0   # ID annotation
start_cat_id = 0   # ID category

def get_all_images(directory):
    valid_exts = ('.jpg', '.jpeg', '.png')
    return {os.path.basename(f): f for root, _, files in os.walk(directory) for f in files if f.lower().endswith(valid_exts)}

def find_best_match(filename, candidates):
    matches = difflib.get_close_matches(filename, candidates, n=1, cutoff=0.3)
    return matches[0] if matches else None


def load_existing_coco(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                data = json.load(f)
                print(f"[INFO] Đã tải file COCO cũ: {path}")
                return data
            except Exception:
                print("[WARN] Old file is error, create new file.")
    return {"images": [], "annotations": [], "categories": []}


def process_one_file(json_file, merged, raw_imgs, annotated_imgs):
    global start_img_id, start_ann_id, start_cat_id

    with open(json_file, "r") as f:
        data = json.load(f)

    id_map = {}

    cat_map = {c["name"]: c["id"] for c in merged["categories"]}
    for cat in data.get("categories", []):
        if cat["name"] not in cat_map:
            start_cat_id += 1
            cat_map[cat["name"]] = start_cat_id
            merged["categories"].append({"id": start_cat_id, "name": cat["name"]})

    for img in data.get("images", []):
        old_name = os.path.basename(img["file_name"])
        new_name = None

        if old_name in annotated_imgs:
            new_name = os.path.relpath(annotated_imgs[old_name], ANNOTATED_DIR)
        else:
            best = find_best_match(old_name, annotated_imgs.keys())
            if best:
                new_name = os.path.relpath(annotated_imgs[best], ANNOTATED_DIR)
            else:
                best_raw = find_best_match(old_name, raw_imgs.keys())
                if best_raw:
                    new_name = os.path.relpath(raw_imgs[best_raw], RAW_DIR)

        if new_name is None:
            print(f"No search have image: {old_name}")
            continue

        start_img_id += 1
        id_map[img["id"]] = start_img_id
        img["id"] = start_img_id
        img["file_name"] = new_name
        merged["images"].append(img)

    for ann in data.get("annotations", []):
        start_ann_id += 1
        ann["id"] = start_ann_id
        ann["image_id"] = id_map.get(ann["image_id"], ann["image_id"])
        ann["category_id"] = cat_map.get(
            next((c["name"] for c in data["categories"] if c["id"] == ann["category_id"]), "unknown"),
            ann["category_id"]
        )
        merged["annotations"].append(ann)


def main():
    raw_imgs = get_all_images(RAW_DIR)
    annotated_imgs = get_all_images(ANNOTATED_DIR)
    json_files = sorted([f for f in os.listdir(COCO_JSON_DIR) if f.endswith(".json")])

    print(f"[INFO] Detected {len(json_files)} file JSON in {COCO_JSON_DIR}")

    merged = load_existing_coco(OUTPUT_PATH)

    for jf in tqdm(json_files, desc="Appending JSONs"):
        path = os.path.join(COCO_JSON_DIR, jf)
        print(f"\n[PROCESSING] {jf}")
        process_one_file(path, merged, raw_imgs, annotated_imgs)

        with open(OUTPUT_PATH, "w") as f:
            json.dump(merged, f, indent=4)

        print(f"[APPENDED] {jf} SUCCESSFULLY")

    print("\n[DONE] All file are saved successfully.")
    print(f"   → Total file: {OUTPUT_PATH}")
    print(f"   → Number of image: {len(merged['images'])}")
    print(f"   → Total annotations: {len(merged['annotations'])}")
    print(f"   → Total categories: {len(merged['categories'])}")


if __name__ == "__main__":
    main()

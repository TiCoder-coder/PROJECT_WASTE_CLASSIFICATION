import os
import re
import sys

source_folder = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/raw"


if not os.path.isdir(source_folder):
    sys.exit(f"No search have source folder: {source_folder}")

files = [
    f for f in os.listdir(source_folder)
    if os.path.isfile(os.path.join(source_folder, f))
]

if not files:
    sys.exit("No search have file to process.")

def extract_number(fname):
    m = re.search(r"\((\d+)\)", fname)
    return int(m.group(1)) if m else 0

files.sort(key=extract_number)

for idx, old_name in enumerate(files, start=1):
    old_path = os.path.join(source_folder, old_name)
    
    _, ext = os.path.splitext(old_name)
    
    new_name = f"image_{idx}{ext}"
    new_path = os.path.join(source_folder, new_name)

    if old_name == new_name:
        print(f"[IGNORE] {old_name} Was existed true name. Continue.")
        continue

    try:
        os.rename(old_path, new_path)
        print(f"Đã xử lý: {old_name} → {new_name}")
    except Exception as e:
        print(f"[LỖI] {old_name}: {e}")

print(f"\n[SUCCESS] ALL FILE WAS RENAMED IN:\n{source_folder}")
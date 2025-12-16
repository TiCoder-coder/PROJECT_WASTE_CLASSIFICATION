import os

root_dir = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/raw"

total_files = 0
total_dirs = 0

for root, dirs, files in os.walk(root_dir):
    total_dirs += len(dirs)
    total_files += len(files)

print(f"Number of child folder: {total_dirs}")
print(f"Number of file: {total_files}")

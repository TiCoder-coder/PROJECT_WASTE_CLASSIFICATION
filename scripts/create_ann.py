import sys
import os
import json
import cv2
import numpy as np
import hydra
from omegaconf import OmegaConf

sys.path.append("/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/sam2")

from sam2.sam2.build_sam import build_sam2
from sam2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
@hydra.main(version_base=None, config_path="../configs/sam2.1", config_name="sam2.1_hiera_l")
def main(cfg):
    ckpt = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/checkpoints/sam2.1_hiera_large.pt"

    model = build_sam2(cfg, ckpt, device="cuda")
    generator = SAM2AutomaticMaskGenerator(model)

    processed_dir = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/processed"
    annotated_dir = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/annotated"

    os.makedirs(annotated_dir, exist_ok=True)

    for root, dirs, files in os.walk(processed_dir):
        for img_file in files:
            if img_file.lower().endswith(('.jpg', '.jpeg')):
                img_path = os.path.join(root, img_file)
                rel_path = os.path.relpath(root, processed_dir)
                ann_subdir = os.path.join(annotated_dir, rel_path)
                os.makedirs(ann_subdir, exist_ok=True)
                ann_file = img_file.replace('.jpg', '_ann.json').replace('.jpeg', '_ann.json')
                ann_path = os.path.join(ann_subdir, ann_file)
                
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                masks = generator.generate(image)
                
                anns = []
                for mask in masks:
                    segmentation = mask['segmentation']
                    contours, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        polygon = contours[0].flatten().tolist()
                        anns.append({"segmentation": polygon})
                
                with open(ann_path, 'w') as f:
                    json.dump(anns, f, indent=2)
                print(f"Created {ann_path} with {len(anns)} annotations")

if __name__ == "__main__":
    main()
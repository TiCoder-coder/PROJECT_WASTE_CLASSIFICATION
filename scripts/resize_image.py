# FILE DUNG DE XU LY BAN DAU: RESIZE ANH VE 1024 x 1024

import os
import cv2
import numpy as np
from typing import List

class Preprocessor:
    def __init__(self, input_dir: str, output_dir: str, target_size: tuple = (1024, 1024)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
        img = cv2.resize(img, self.target_size)
        img = img.astype(np.float32) / 255.0
        return img

    def save_processed(self, img: np.ndarray, output_path: str):
        img_save = (img * 255).astype(np.uint8)
        cv2.imwrite(output_path, img_save)

    def run(self):
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(root, file)

                    try:
                        processed_img = self.preprocess_image(img_path)

                        out_path = os.path.join(self.output_dir, file)

                        self.save_processed(processed_img, out_path)
                        print(f"Processed: {img_path} -> {out_path}")

                    except Exception as e:
                        print(f"Error preprocessing {img_path}: {e}")

if __name__ == "__main__":
    preprocessor = Preprocessor(
        input_dir="/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/raw",
        output_dir="/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/resize"
    )
    preprocessor.run()

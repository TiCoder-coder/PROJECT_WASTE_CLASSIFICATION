import torch
import numpy as np

PTH_PATH = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/output/sam2_finetuned_final.pth"


def separator():
    print("\n" + "-"*80 + "\n")


def print_tensor_info(name, tensor):
    if isinstance(tensor, torch.Tensor):
        print(f"{name}: Tensor | shape={tuple(tensor.shape)} | dtype={tensor.dtype} | numel={tensor.numel()}")
    elif isinstance(tensor, np.ndarray):
        print(f"{name}: Numpy Array | shape={tensor.shape} | dtype={tensor.dtype}")
    else:
        print(f"{name}: {type(tensor)}")


def inspect_pth(path):

    separator()
    print(f"[INFO] Loading checkpoint from: {path}")
    separator()

    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        print("[TYPE] state_dict (raw PyTorch weights)")
        separator()

        total_params = 0
        for k, v in ckpt.items():
            print_tensor_info(k, v)
            total_params += v.numel()

        separator()
        print(f"[SUMMARY] Total parameters: {total_params:,}")
        return

    if isinstance(ckpt, dict):
        print("[TYPE] checkpoint dict (multiple components)")
        separator()

        print(f"[INFO] Keys in checkpoint: {list(ckpt.keys())}")
        separator()

        if "classes" in ckpt:
            print("[CLASSES]", ckpt["classes"])
            separator()

        if "epoch" in ckpt:
            print(f"[EPOCH] {ckpt['epoch']}")
            separator()

        if "optimizer" in ckpt:
            print("[OPTIMIZER] Found optimizer state dict")
            print(f" - keys: {list(ckpt['optimizer'].keys())}")
            separator()

        if "model_state_dict" in ckpt:
            print("[MODEL_STATE_DICT] Inspecting weights...")
            separator()
            sd = ckpt["model_state_dict"]

            total_params = 0
            for k, v in sd.items():
                print_tensor_info(k, v)
                total_params += v.numel()

            separator()
            print(f"[SUMMARY] Total model params: {total_params:,}")
            return

        print("[WARN] Unknown dictionary structure → listing contents")
        for k, v in ckpt.items():
            print(f"{k}: {type(v)}")
        return

    print("[TYPE] Full model object")
    separator()
    print(ckpt)
    print("\n[WARN] Full model loaded. Cannot inspect layers without model class definition.")


if __name__ == "__main__":
    inspect_pth(PTH_PATH)

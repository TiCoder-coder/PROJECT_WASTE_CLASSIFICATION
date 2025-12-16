import os, cv2, torch, random
import numpy as np
import sys
from ultralytics import YOLO
from collections import defaultdict
from torchvision import models
import torchvision.transforms as T
BASE_DIR = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT"

PROJECT_ROOT = BASE_DIR
SAM2_ROOT = os.path.join(PROJECT_ROOT, "sam2")
for p in [SAM2_ROOT, os.path.join(SAM2_ROOT, "sam2")]:
    if p not in sys.path: sys.path.insert(0, p)

from sam2_image_predictor import SAM2ImagePredictor
from build_sam import build_sam2

device = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_CKPT = os.path.join(BASE_DIR, "checkpoints", "yolo11n.pt")
SAM2_CKPT = os.path.join(BASE_DIR, "output", "sam2_finetuned_final.pth")
SAM2_CFG  = os.path.join(BASE_DIR, "configs/sam2.1/sam2.1_hiera_b+.yaml")

# ---------------- YOLO INIT ----------------
yolo = YOLO(YOLO_CKPT).to(device)

# Random color fixed per class
random.seed(99)
CLS_COLOR = {
    name: (
        random.randint(80,255),
        random.randint(80,255),
        random.randint(80,255)
    )
    for name in yolo.model.names.values()
}

# ---------------- SAM2 INIT ----------------
sd = torch.load(SAM2_CKPT, map_location="cpu")
model = build_sam2(SAM2_CFG, None, device)
model.load_state_dict(sd, strict=False)
model = model.to(device).eval()
predictor = SAM2ImagePredictor(model)

# Preprocess for embeddings
backbone = models.resnet18(pretrained=True)
backbone = torch.nn.Sequential(*list(backbone.children())[:-1]).to(device).eval()

preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# ---------------- TRACKER ----------------
class SimpleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, dets):
        results = []
        for d in dets:
            d["id"] = self.next_id
            self.next_id += 1
            results.append(d)
        return results

tracker = SimpleTracker()


# ---------------- MAIN INFERENCE ----------------
def run_detection(frame):
    """Nhận frame (numpy BGR) → trả về list detections cho FE"""

    # YOLO detect
    results = yolo.predict(frame, conf=0.55, verbose=False)
    boxes, cls_ids, confs = [], [], []

    for r in results:
        for b, c, cls in zip(r.boxes.xyxy.cpu().numpy(),
                             r.boxes.conf.cpu().numpy(),
                             r.boxes.cls.cpu().numpy()):
            boxes.append(tuple(map(int, b)))
            confs.append(float(c))
            cls_ids.append(int(cls))

    if len(boxes) == 0:
        return []

    # SAM2 mask
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    boxes_np = np.array(boxes, dtype=np.int32)

    masks, scores, _ = predictor.predict(box=boxes_np, multimask_output=False)
    if masks.ndim == 4: 
        masks = masks[:,0]

    H, W = frame.shape[:2]
    detections = []

    for i, (box, mask) in enumerate(zip(boxes, masks)):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

        cls_name = yolo.model.names[cls_ids[i]]
        color = CLS_COLOR[cls_name]

        detections.append({
            "id": i,
            "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            "cls": cls_name,
            "score": confs[i],
            "mask": mask.tolist(),   # gửi mask về FE
            "color": color
        })

    return detections

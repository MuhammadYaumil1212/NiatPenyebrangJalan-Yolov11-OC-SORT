import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models
from ultralytics import YOLO

from ocsort_tracker.ocsort import OCSort

print("Loading YOLO Model...")
# Inisialisasi model dan tracker
modelYolo11 = YOLO("../yolo-weights/yolo11l.pt")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_path = "yolo_nas_pose_l_coco_pose.pth"
input_path = "videos_annotated"
yolo_nas = models.get(
    Models.YOLO_NAS_POSE_L,
    num_classes=17,
    checkpoint_path=model_path
).to(device)
tracker = OCSort(det_thresh=0.15, max_age=30, min_hits=1)

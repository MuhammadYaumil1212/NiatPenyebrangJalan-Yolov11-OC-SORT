from ocsort_tracker.ocsort import OCSort
from ultralytics import YOLO
import xml.etree.ElementTree as ET
import cv2
import cvzone
import numpy as np
import torch
import os

input_folder = "videos"
xml_folder = "annotations"
output_folder = "videos_annotated"

os.makedirs(output_folder, exist_ok=True)

# Load Model
print("Loading YOLO Model...")
model = YOLO("../yolo-weights/yolo11l.pt")

all_files = os.listdir(input_folder)
video_files = [f for f in all_files if f.endswith('.mp4')]
video_files.sort()

print(f"Ditemukan {len(video_files)} video. Mode: HEADLESS (Tanpa Tampilan).")

for idx_video, filename in enumerate(video_files):
    input_video_path = os.path.join(input_folder, filename)
    xml_filename = filename.replace('.mp4', '.xml')
    xml_path = os.path.join(xml_folder, xml_filename)

    output_filename = filename.replace('.mp4', '_annotated.mp4')
    output_path = os.path.join(output_folder, output_filename)

    if not os.path.exists(xml_path):
        print(f"[SKIP] XML tidak ada: {filename}")
        continue

    print(f"\n[{idx_video + 1}/{len(video_files)}] Memproses: {filename}...")

    cap = cv2.VideoCapture(input_video_path)

    # Ambil total frame untuk estimasi progress
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML: {e}")
        continue

    tracker = OCSort(det_thresh=0.15, max_age=30, min_hits=1)
    frame_count = 0
    gt_crossing_state = {}

    while True:
        success, img = cap.read()
        if not success:
            break

        # Print update setiap 50 frame agar terminal tidak penuh
        if frame_count % 50 == 0:
            print(f"   > Frame {frame_count}/{total_frames} processed...", end='\r')

        gt_boxes = []
        for idx_track, track in enumerate(root.findall('track')):
            gt_box = track.find(f"./box[@frame='{frame_count}']")
            if gt_box is None: continue
            track_id = None
            id_attr = gt_box.find("./attribute[@name='id']")
            if id_attr is not None:
                track_id = id_attr.text
            elif track.get('id') is not None:
                track_id = track.get('id')
            else:
                track_id = str(idx_track)

            x1 = float(gt_box.get('xtl'))
            y1 = float(gt_box.get('ytl'))
            x2 = float(gt_box.get('xbr'))
            y2 = float(gt_box.get('ybr'))

            cross_attr = gt_box.find("./attribute[@name='cross']")
            is_crossing = False
            if cross_attr is not None:
                is_crossing = (cross_attr.text == 'crossing')
                gt_crossing_state[track_id] = is_crossing
            else:
                is_crossing = gt_crossing_state.get(track_id, False)

            gt_boxes.append([x1, y1, x2, y2, is_crossing])

        # Gambar Ground Truth
        for gx1, gy1, gx2, gy2, crossing in gt_boxes:
            color = (0, 255, 0) if crossing else (0, 0, 255)
            cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), color, 2)
            label = "Crossing" if crossing else "Not Crossing"
            cvzone.putTextRect(img, label, (int(gx1), max(30, int(gy1) - 10)), scale=1, thickness=1, offset=3,
                               colorR=color)

        # YOLO & Tracker
        detections = []
        results = model(img, conf=0.15, iou=0.5, verbose=False)

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                    conf = float(box.conf[0])
                    detections.append([x1, y1, x2, y2, conf, 0])

        if len(detections) > 0:
            detections = torch.from_numpy(np.array(detections, dtype=np.float32))
        else:
            detections = torch.empty((0, 6))

        h, w, _ = img.shape
        tracks = tracker.update(detections, (h, w), (h, w))

        for t in tracks:
            tx1, ty1, tx2, ty2, track_id_tracker = map(int, t[:5])
            w_box = tx2 - tx1
            h_box = ty2 - ty1
            cvzone.cornerRect(img, bbox=(tx1, ty1, w_box, h_box), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f'YOLO: {track_id_tracker}', (tx1, max(35, ty1 + 20)), scale=1, thickness=1,
                               offset=3, colorR=(255, 0, 255))

        out.write(img)
        # cv2.imshow(...)
        # cv2.waitKey(...)
        frame_count += 1

    cap.release()
    out.release()
    print(f"   > Selesai! Disimpan ke: {output_path}")

cv2.destroyAllWindows()
print("\nSemua tugas selesai!")
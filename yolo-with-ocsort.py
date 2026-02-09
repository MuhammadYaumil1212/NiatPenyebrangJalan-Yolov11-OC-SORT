import xml.etree.ElementTree as ET
import cv2
import cvzone
import numpy as np
import os

# --- KONFIGURASI FOLDER ---
input_folder = "videos"
xml_folder = "annotations"
output_folder = "videos_annotated"

os.makedirs(output_folder, exist_ok=True)

# Ambil daftar file
all_files = os.listdir(input_folder)
video_files = [f for f in all_files if f.endswith('.mp4')]
video_files.sort()

print(f"Ditemukan {len(video_files)} video. Mode: HEADLESS (Ground Truth Only).")

# --- MULAI PROSES ---
for idx_video, filename in enumerate(video_files):
    input_video_path = os.path.join(input_folder, filename)
    xml_filename = filename.replace('.mp4', '.xml')
    xml_path = os.path.join(xml_folder, xml_filename)

    output_filename = filename.replace('.mp4', '_gt_only.mp4')
    output_path = os.path.join(output_folder, output_filename)

    if not os.path.exists(xml_path):
        print(f"[SKIP] XML tidak ada: {filename}")
        continue

    print(f"\n[{idx_video + 1}/{len(video_files)}] Memproses: {filename}...")

    cap = cv2.VideoCapture(input_video_path)

    # Properti Video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Parse XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML: {e}")
        continue

    frame_count = 0
    gt_crossing_state = {}

    while True:
        success, img = cap.read()
        if not success:
            break

        # Progress bar di terminal
        if frame_count % 50 == 0:
            print(f"   > Frame {frame_count}/{total_frames} processed...", end='\r')

        gt_boxes = []

        # --- PARSING XML (LOGIKA ID YANG SUDAH DIPERBAIKI) ---
        for idx_track, track in enumerate(root.findall('track')):
            gt_box = track.find(f"./box[@frame='{frame_count}']")
            if gt_box is None: continue

            # 1. Cari ID di dalam atribut box
            track_id = None
            id_attr = gt_box.find("./attribute[@name='id']")

            if id_attr is not None:
                track_id = id_attr.text
            elif track.get('id') is not None:
                # 2. Cari di header track
                track_id = track.get('id')
            else:
                # 3. Fallback ke index loop
                track_id = str(idx_track)

            x1 = float(gt_box.get('xtl'))
            y1 = float(gt_box.get('ytl'))
            x2 = float(gt_box.get('xbr'))
            y2 = float(gt_box.get('ybr'))

            # Cek status Crossing
            cross_attr = gt_box.find("./attribute[@name='cross']")
            is_crossing = False

            if cross_attr is not None:
                # Sesuaikan dengan nilai di XML kamu (crossing / not-crossing)
                is_crossing = (cross_attr.text == 'crossing')
                gt_crossing_state[track_id] = is_crossing
            else:
                # Pakai memori terakhir
                is_crossing = gt_crossing_state.get(track_id, False)

            gt_boxes.append([x1, y1, x2, y2, is_crossing])

        # --- GAMBAR KOTAK GROUND TRUTH ---
        for gx1, gy1, gx2, gy2, crossing in gt_boxes:
            # Hijau = Crossing, Merah = Not Crossing
            color = (0, 255, 0) if crossing else (0, 0, 255)

            cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), color, 2)

            label = "Crossing" if crossing else "Not Crossing"
            cvzone.putTextRect(img, label, (int(gx1), max(30, int(gy1) - 10)),
                               scale=1, thickness=1, offset=3, colorR=color)

        # --- SIMPAN FRAME ---
        out.write(img)
        frame_count += 1

    # Cleanup per video
    cap.release()
    out.release()
    print(f"   > Selesai! Disimpan ke: {output_path}")

cv2.destroyAllWindows()
print("\nSemua tugas selesai!")
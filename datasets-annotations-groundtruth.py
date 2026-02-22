import os
import random
import xml.etree.ElementTree as ET

import cv2
import cvzone

input_folder = "original_videos"
xml_folder = "annotations"
output_folder = "annotated_videos_output"  # --- BARU: Tentukan nama folder hasil save ---

# --- BARU: Buat folder output secara otomatis jika belum ada ---
os.makedirs(output_folder, exist_ok=True)

label_colors = {}


def get_color_for_label(label):
    """Fungsi untuk menghasilkan dan menyimpan warna unik untuk setiap kelas label."""
    if label not in label_colors:
        # Generate warna BGR acak (hindari warna terlalu gelap agar teks terbaca)
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        label_colors[label] = color
    return label_colors[label]


# Ambil semua file video
all_files = os.listdir(input_folder)
video_files = [f for f in all_files if f.endswith('.mp4')]
video_files.sort()

for idx_video, filename in enumerate(video_files):

    input_video_path = os.path.join(input_folder, filename)
    xml_filename = filename.replace('.mp4', '.xml')
    xml_path = os.path.join(xml_folder, xml_filename)

    output_video_path = os.path.join(output_folder, f"annotated_{filename}")

    if not os.path.exists(xml_path):
        print(f"XML tidak ada: {filename}")
        continue

    print(f"\n[{idx_video + 1}/{len(video_files)}] Memproses: {filename}...")

    cap = cv2.VideoCapture(input_video_path)

    # Ambil metadata video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Gunakan codec 'mp4v' untuk membuat file .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML: {e}")
        cap.release()
        out.release()
        continue

    # Inisialisasi penghitung frame
    frame_count = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # Cetak progres setiap 50 frame agar terminal tidak penuh
        if frame_count % 50 == 0:
            print(f"   > Frame {frame_count}/{total_frames} processed...", end='\r')

        # Proses ground truth
        gt_boxes = []
        for idx_track, track in enumerate(root.findall('track')):
            gt_box = track.find(f"./box[@frame='{frame_count}']")
            if gt_box is None:
                continue

            # Ambil label dinamis langsung dari tag <track>
            obj_label = track.get('label', 'Unknown')

            # Ambil koordinat
            x1 = float(gt_box.get('xtl'))
            y1 = float(gt_box.get('ytl'))
            x2 = float(gt_box.get('xbr'))
            y2 = float(gt_box.get('ybr'))

            gt_boxes.append([x1, y1, x2, y2, obj_label])

        # Gambar Ground Truth ke atas frame
        for gx1, gy1, gx2, gy2, label_text in gt_boxes:
            # Dapatkan warna yang sesuai dengan label teks
            color = get_color_for_label(label_text)

            cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), color, 2)
            cvzone.putTextRect(img, str(label_text), (int(gx1), max(30, int(gy1) - 10)),
                               scale=1, thickness=1, offset=3, colorR=color)
        out.write(img)
        cv2.imshow("Ground Truth Viewer", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print("\n\nDihentikan paksa oleh pengguna.")
            exit()

        frame_count += 1
    cap.release()
    out.release()

cv2.destroyAllWindows()
print(f"\nSemua tugas selesai! Video hasil anotasi tersimpan di folder: {output_folder}")

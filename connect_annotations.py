import cv2
import xml.etree.ElementTree as ET


def connect_annotations(video_path, xml_path):
    # 1. Load Video
    cap = cv2.VideoCapture(video_path)

    # 2. Load & Parse XML (Ground Truth)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        for track in root.findall('track'):
            ped_id = track.get('id')
            label = track.get('label')

            # Cari box spesifik untuk frame ini
            box = track.find(f"./box[@frame='{frame_no}']")

            if box is not None:
                # Ambil koordinat (xtl: x top-left, ytl: y top-left, dst)
                xtl = int(float(box.get('xtl')))
                ytl = int(float(box.get('ytl')))
                xbr = int(float(box.get('xbr')))
                ybr = int(float(box.get('ybr')))

                # Cek status 'crossing'
                is_crossing = box.find("./attribute[@name='crossing']")
                status = "Crossing" if (is_crossing is not None and is_crossing.text == 'true') else "Not Crossing"

                color = (0, 255, 0) if status == "Crossing" else (0, 0, 255)
                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), color, 2)
                cv2.putText(frame, f"ID:{ped_id} {status}", (xtl, ytl - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Tampilkan hasil penggabungan
        cv2.imshow('JAAD GT Connected', frame)

        frame_no += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Jalankan dengan path file Anda
# connect_xml_to_video('video_0001.mp4', 'video_0001.xml')
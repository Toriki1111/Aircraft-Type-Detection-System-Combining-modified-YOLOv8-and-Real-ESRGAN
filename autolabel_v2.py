import os
import shutil
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# 1. Khai báo file CSV và đường dẫn
CSV_FILES = {
    "train": "dataset/train.csv",
    "val": "dataset/val.csv",
    "test": "dataset/test.csv"
}

SRC_IMAGE_DIR = "dataset/fgvc-aircraft-2013b/data/images" # Thư mục chứa ảnh gốc vừa tải
OUTPUT_DIR = "dataset_v2" # Output folder mới

detector = YOLO("yolov8l.pt")

print(f"🚀 Bắt đầu dùng 'yolov8l.pt' để Auto-Label vị trí máy bay -> Thư mục '{OUTPUT_DIR}'...\n")

for split, csv_path in CSV_FILES.items():
    if not os.path.exists(csv_path):
        print(f"⚠️ Không tìm thấy {csv_path}, bỏ qua...")
        continue

    df = pd.read_csv(csv_path)

    out_img_dir = os.path.join(OUTPUT_DIR, split, "images")
    out_lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    success_count = 0

    for idx, row in df.iterrows():
        img_name = str(row['filename']).strip()
        class_id = int(row['Labels']) # ID chuẩn từ CSV (0-33 cho V2)

        src_img_path = os.path.join(SRC_IMAGE_DIR, img_name)
        dst_img_path = os.path.join(out_img_dir, img_name)

        txt_name = os.path.splitext(img_name)[0] + ".txt"
        dst_txt_path = os.path.join(out_lbl_dir, txt_name)

        if not os.path.exists(src_img_path):
            continue

        # Copy ảnh sang folder dataset_v2
        shutil.copy(src_img_path, dst_img_path)

        # Lấy kích thước ảnh
        with Image.open(src_img_path) as img:
            img_w, img_h = img.size

        # Dùng YOLOv8l detect vị trí khung máy bay
        results = detector.predict(src_img_path, verbose=False, conf=0.25)

        boxes_lines = []
        for result in results:
            for box in result.boxes:
                # Nếu phát hiện là airplane (COCO class 4) hoặc lấy box tự tin nhất
                if int(box.cls[0]) == 4 or len(result.boxes) == 1:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    cx = round(((x1 + x2) / 2.0) / img_w, 6)
                    cy = round(((y1 + y2) / 2.0) / img_h, 6)
                    bw = round((x2 - x1) / img_w, 6)
                    bh = round((y2 - y1) / img_h, 6)

                    boxes_lines.append(f"{class_id} {cx} {cy} {bw} {bh}\n")
                    break

        # Fallback full-box nếu detector không tìm thấy
        if not boxes_lines:
            boxes_lines.append(f"{class_id} 0.5 0.5 1.0 1.0\n")

        with open(dst_txt_path, "w", encoding="utf-8") as f:
            f.writelines(boxes_lines)

        success_count += 1

    print(f"✅ [{split.upper()}] Đã gán nhãn xong {success_count} ảnh -> {OUTPUT_DIR}/{split}/")

print(f"\n🎉 HOÀN THÀNH AUTO-LABEL! Thư mục '{OUTPUT_DIR}' đã sẵn sàng!")

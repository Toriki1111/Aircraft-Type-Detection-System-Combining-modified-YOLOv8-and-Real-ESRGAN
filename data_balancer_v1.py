import os
import shutil
from PIL import Image


RAW_DATA_DIR = "dataset/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data"
OUTPUT_YOLO_DIR = "dataset/3_dataset_yolo"

CLASSES = {"Airbus": 0, "Boeing": 1}

def load_labels(file_path):
    """Đọc file ánh xạ giữa Tên ảnh và Hãng sản xuất (Airbus/Boeing)"""
    labels_dict = {}
    if not os.path.exists(file_path):
        return labels_dict
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                img_id, manufacturer = parts[0], parts[1]
                if manufacturer in CLASSES:
                    labels_dict[img_id] = manufacturer
    return labels_dict

def load_bboxes(file_path):
    """Đọc file images_box.txt chứa tọa độ khung hình chữ nhật gốc (Xmin Ymin Xmax Ymax)"""
    bboxes_dict = {}
    if not os.path.exists(file_path):
        return bboxes_dict
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                img_id = parts[0]
                # Tọa độ gốc của bộ dữ liệu dạng pixels: x_min, y_min, x_max, y_max
                bbox = [float(x) for x in parts[1:]]
                bboxes_dict[img_id] = bbox
    return bboxes_dict

def convert_to_yolo_format(img_width, img_height, bbox):
    """Chuyển đổi tọa độ pixel (DMS/BBox gốc) sang định dạng Normalization (0-1) chuẩn của YOLO"""
    x_min, y_min, x_max, y_max = bbox
    
    # Tính toán tâm, chiều rộng, chiều cao của khung hình chữ nhật
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_center = x_min + (box_width / 2.0)
    y_center = y_min + (box_height / 2.0)
    
    # Chuẩn hóa về khoảng từ 0 đến 1 bằng cách chia cho kích thước ảnh
    src_x = x_center / img_width
    src_y = y_center / img_height
    src_w = box_width / img_width
    src_h = box_height / img_height
    
    return src_x, src_y, src_w, src_h

def process_dataset_split(split_name, label_file_name, bboxes_dict):
    """Xử lý tự động gom ảnh, chuyển nhãn và phân chia tập dữ liệu Train/Val"""
    print(f"🔄 Đang xử lý tập dữ liệu: {split_name}...")
    
    label_file_path = os.path.join(RAW_DATA_DIR, label_file_name)
    img_labels = load_labels(label_file_path)
    
    # Tạo các thư mục đích chuẩn YOLO nếu chưa tồn tại
    dest_img_dir = os.path.join(OUTPUT_YOLO_DIR, split_name, "images")
    dest_lbl_dir = os.path.join(OUTPUT_YOLO_DIR, split_name, "labels")
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_lbl_dir, exist_ok=True)
    
    count = 0
    for img_id, manufacturer in img_labels.items():
        img_name = f"{img_id}.jpg"
        src_img_path = os.path.join(RAW_DATA_DIR, "images", img_name)
        
        # Kiểm tra xem ảnh và tọa độ khung tương ứng có tồn tại không
        if os.path.exists(src_img_path) and img_id in bboxes_dict:
            try:
                # Mở ảnh để lấy kích thước chiều rộng và chiều cao thực tế
                with Image.open(src_img_path) as img:
                    width, height = img.size
                
                # Chuyển đổi nhãn sang chuẩn YOLO
                bbox = bboxes_dict[img_id]
                yolo_box = convert_to_yolo_format(width, height, bbox)
                class_id = CLASSES[manufacturer]
                
                # 1. Copy file ảnh sang thư mục đích sạch
                shutil.copy(src_img_path, os.path.join(dest_img_dir, img_name))
                
                # 2. Tạo file text nhãn .txt tương ứng cho YOLO
                lbl_name = f"{img_id}.txt"
                with open(os.path.join(dest_lbl_dir, lbl_name), 'w', encoding='utf-8') as f_lbl:
                    f_lbl.write(f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")
                
                count += 1
            except Exception as e:
                print(f"Lỗi xử lý ảnh {img_name}: {e}")
                
    print(f"✅ Đã xử lý xong {count} ảnh cho tập {split_name} (Hãng: Airbus & Boeing).")

if __name__ == "__main__":
    # Tải toàn bộ tọa độ bounding box vào bộ nhớ
    bbox_path = os.path.join(RAW_DATA_DIR, "images_box.txt")
    if not os.path.exists(bbox_path):
        print("Không tìm thấy file images_box.txt! Hãy đảm bảo đống file thô nằm đúng trong dataset/3_dataset_yolo/")
    else:
        all_bboxes = load_bboxes(bbox_path)
        
        # Chạy quy trình tự động bóc tách cho cả 2 tập huấn luyện (train) và kiểm thử (val)
        process_dataset_split("train", "images_manufacturer_train.txt", all_bboxes)
        process_dataset_split("val", "images_manufacturer_val.txt", all_bboxes)
        print("\n🎉 HOÀN THÀNH PIPELINE DỮ LIỆU! Thư mục data/3_dataset_yolo đã sẵn sàng để train AI!")
from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("webapp/best_v1.pt")
    
    results = model.val(
        data="dataset_v1.yml",  # Đường dẫn tới file data.yaml của dự án
        split="test",  # Kích hoạt đánh giá trên tập Test
        imgsz=640,  # Kích thước ảnh đánh giá
        batch=16,
        project="runs/test_results",  # Thư mục lưu kết quả
        name="test_airbus_boeing",  # Tên folder kết quả
        save_json=True,  # Lưu lại JSON metrics nếu cần
        plots=True,  # Auto making PR-curve, Confusion Matrix...
    )

    print("\n" + "=" * 50)
    print("📊 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print("=" * 50)
    print(f" Precision (Độ chính xác): {results.results_dict['metrics/precision(B)']:.4f}")
    print(f" Recall (Độ bao phủ):    {results.results_dict['metrics/recall(B)']:.4f}")
    print(f" mAP50:                 {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f" mAP50-95:              {results.results_dict['metrics/mAP50-95(B)']:.4f}")
    print("=" * 50)

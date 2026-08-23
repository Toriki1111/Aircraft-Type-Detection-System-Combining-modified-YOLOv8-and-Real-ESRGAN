from ultralytics import YOLO

if __name__ == "__main__":
    # 1. Load file mô hình best.pt đã train xong của bạn
    model = YOLO("webapp/best_v2.pt")  # Điền đúng đường dẫn file best.pt của bạn

    # 2. Đánh giá trên tập TEST
    # split='test' ép YOLO dùng đường dẫn test: trong file data.yaml
    results = model.val(
        data="dataset_v2_variant.yml",  # Đường dẫn tới file data.yaml của dự án
        split="test",  # Kích hoạt đánh giá trên tập Test
        imgsz=640,  # Kích thước ảnh đánh giá
        batch=16,
        project="runs/test_results",  # Thư mục lưu kết quả
        name="test_variant_plane",  # Tên folder kết quả
        save_json=True,  # Lưu lại JSON metrics nếu cần
        plots=True,  # Tự động vẽ các biểu đồ PR-curve, Confusion Matrix...
    )

    # 3. In các chỉ số quan trọng ra màn hình Terminal
    print("\n" + "=" * 50)
    print("📊 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print("=" * 50)
    print(f" Precision (Độ chính xác): {results.results_dict['metrics/precision(B)']:.4f}")
    print(f" Recall (Độ bao phủ):    {results.results_dict['metrics/recall(B)']:.4f}")
    print(f" mAP50:                 {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f" mAP50-95:              {results.results_dict['metrics/mAP50-95(B)']:.4f}")
    print("=" * 50)
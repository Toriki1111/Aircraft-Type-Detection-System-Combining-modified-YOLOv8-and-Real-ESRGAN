from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8l.pt")
    results = model.train(
        data='dataset_v1.yaml',       
        epochs=100,                # Tăng lên 100 epochs để AI học kỹ hơn, không lo thiếu thời gian
        imgsz=800,                 # Tăng kích thước ảnh lên 800px để giữ lại độ nét chi tiết của máy bay
        batch=16,                  # Nếu card RTX có VRAM từ 8GB-12GB trở lên thì để 16 hoặc 32 chạy cực nhanh
        device=0,                  
        workers=4,                 # Tăng số luồng CPU lên 4 hoặc 8 để nạp ảnh vào GPU nhanh hơn
        project='runs/train',      
        name='airbus_boeing_heavy_model' 
    )
    
    print("\n HUÂN LUYỆN V1 HOÀN TẤT!")

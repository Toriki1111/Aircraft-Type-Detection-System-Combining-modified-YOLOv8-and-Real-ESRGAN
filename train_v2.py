from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8l.pt")

    results = model.train(
        data="dataset_v2_variant.yml",
        epochs=80,                
        imgsz=800,                 
        batch=8,                  
        device=0,                  
        workers=8,                 
        project='runs/train',      
        name='variant_airbus_boeing_v2',
    )
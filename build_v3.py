import os
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 1. Cấu hình đường dẫn
LOGO_DIR = os.path.join(os.getcwd(), 'logo')
WEIGHTS_DIR = os.path.join(os.getcwd(), 'weights')
OUTPUT_PKL = os.path.join(WEIGHTS_DIR, 'v3_logo_features.pkl')

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# 2. Khởi tạo mô hình trích xuất đặc trưng (ResNet18 Pretrained)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Đang chạy trên thiết bị: {device}")

# Lấy ResNet18 và bỏ lớp Classification cuối cùng đi (chỉ giữ phần trích xuất feature)
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)
feature_extractor.eval()

# Transform ảnh chuẩn hóa cho ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_vector(image_path):
    """Trích xuất vector 512 chiều từ 1 bức ảnh logo"""
    try:
        img = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = feature_extractor(tensor) # (1, 512, 1, 1)
            embedding = torch.flatten(embedding, 1) # (1, 512)
            # Chuẩn hóa L2 norm để tính Cosine Similarity siêu nhanh
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(0).cpu().numpy()
    except Exception as e:
        print(f"❌ Lỗi khi đọc ảnh {image_path}: {e}")
        return None

# 3. Quét toàn bộ folder logo/ và lưu vào Dictionary
logo_database = {}
image_extensions = ('.png', '.jpg', '.jpeg', '.webp')

print("🔍 Đang tiến hành nén đống logo thành Vector đặc trưng V3...")
for file in os.listdir(LOGO_DIR):
    if file.lower().endswith(image_extensions):
        img_path = os.path.join(LOGO_DIR, file)
        
        # Lấy tên hãng hàng không từ tên file (bỏ đuôi .png)
        airline_name = os.path.splitext(file)[0].replace('_', ' ')
        
        vector = extract_vector(img_path)
        if vector is not None:
            logo_database[airline_name] = vector
            print(f"  ---> Đã xử lý: {airline_name}")

# 4. Lưu kết quả ra file .pkl
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(logo_database, f)

print(f"\n🎉 THÀNH CÔNG! Đã tạo file V3 tại: {OUTPUT_PKL}")
print(f"📊 Tổng cộng đã học được {len(logo_database)} logo hãng hàng không!")
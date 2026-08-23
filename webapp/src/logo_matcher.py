import os
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class LogoMatcher:
    def __init__(self, pkl_path='weights/v3_logo_features.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load mô hình ResNet18 trích xuất đặc trưng
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.feature_extractor.eval()
        
        # Transform chuẩn hóa ảnh
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load CSDL vector logo
        self.logo_database = {}
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self.logo_database = pickle.load(f)

    def extract_vector(self, pil_img):
        """Trích xuất vector 512D từ PIL Image"""
        try:
            img = pil_img.convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.feature_extractor(tensor)
                embedding = torch.flatten(embedding, 1)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.squeeze(0).cpu().numpy()
        except Exception:
            return None

    def predict_airline_from_crop(self, crop_img, threshold=0.5):
        """So sánh Cosine Similarity vùng ảnh cắt (crop) với CSDL Logo"""
        if not self.logo_database:
            return "Unknown", 0.0
            
        crop_vector = self.extract_vector(crop_img)
        if crop_vector is None:
            return "Unknown", 0.0

        best_airline = "Unknown"
        max_sim = -1.0

        for airline_name, db_vector in self.logo_database.items():
            sim = float(np.dot(crop_vector, db_vector)) # Cosine similarity
            if sim > max_sim:
                max_sim = sim
                best_airline = airline_name

        if max_sim >= threshold:
            return best_airline, max_sim
        return "Unknown", max_sim
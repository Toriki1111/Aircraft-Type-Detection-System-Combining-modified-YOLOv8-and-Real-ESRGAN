# ✈️ Single-Shot Flight Lookup System

> **A Hybrid Deep Learning System combining Modified YOLOv8, Real-ESRGAN, and ResNet18 Feature Matching for Aircraft Model Identification & Airline Verification.**

---

## 📌 Features

* **Real-ESRGAN Integration**: Super-resolution ($x4$) pre-processing to restore blurry or low-resolution images.
* **YOLOv8 Detection**: Custom-trained YOLOv8 (`best_v2.pt`) for accurate aircraft detection and model classification.
* **ResNet18 Logo Matching**: Fallback feature-extraction pipeline using Cosine Similarity to verify airline logos independently.
* **Offline-capable & Real-time UI**: Built-in Streamlit web app providing instant flight lookup and metadata aggregation.

---

## 🛠️ Project Structure

```text
Do-an-nganh/
├── logo/                   # Reference airline logo database
├── webapp/                 # Streamlit application source
│   ├── src/
│   │   ├── image_processor.py  # Real-ESRGAN handler
│   │   ├── logo_matcher.py     # ResNet18 vector matching
│   │   ├── metadata_worker.py  # Flight details processing
│   │   └── radar_api.py        # Radar / Flight lookup service
│   ├── app.py                  # Main Streamlit web application
│   └── RealESRGAN_x4plus.pth   # Pre-trained Real-ESRGAN weights
├── best_v2.pt              # Best trained YOLOv8 model weights
├── requirements.txt        # Python dependencies
└── README.md
```
1. Installation & Setup
git clone [https://github.com/Toriki1111/Research-and-Development-of-Aircraft-Type-Detection-System-Combining-modified-YOLOv8-and-Real-ESRGAN.git](https://github.com/Toriki1111/Research-and-Development-of-Aircraft-Type-Detection-System-Combining-modified-YOLOv8-and-Real-ESRGAN.git)
cd Research-and-Development-of-Aircraft-Type-Detection-System-Combining-modified-YOLOv8-and-Real-ESRGAN

Cài PyTorch (Có hỗ trợ CUDA / GPU)

Bash
```text
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
(Nếu máy đó chỉ chạy CPU, đổi cu121 thành cpu)
```
2. Set Up Virtual Environment
# Windows (PowerShell)
```text
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
# Linux / macOS
```text
python3 -m venv .venv
source .venv/bin/activate
```
3. Install Dependencies
```text
pip install --upgrade pip
pip install -r requirements.txt 
```
💻 Running the Application
Launch the Streamlit web interface using:

Bash
```text
streamlit run webapp/app.py
```
🛠️ Tech Stack
Language: Python 3.10+

Frameworks & Libraries: PyTorch, Ultralytics YOLOv8, OpenCV, Pillow, Streamlit

Pre-trained Weights: Real-ESRGAN (RealESRGAN_x4plus.pth), ResNet18 (Torchvision)
---

Lỗi thường gặp khi cài:
```text
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```
truy cap vo: .venv\Lib\site-packages\basicsr\data\degradations.py (VScode,app bạn xài,...)
```text
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```
đổi thành: 
```text
from torchvision.transforms.functional import rgb_to_grayscale (Chỉ đơn giản là bỏ chữ _tensor đi là được)
```




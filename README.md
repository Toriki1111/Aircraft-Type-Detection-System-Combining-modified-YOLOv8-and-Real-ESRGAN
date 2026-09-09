# ✈️ Single-Shot Flight Lookup System

> **A Hybrid Deep Learning System combining Modified YOLOv8, Real-ESRGAN, and ResNet18 Feature Matching for Aircraft Model Identification & Airline Verification.**

---

## 📌 Features

* **Real-ESRGAN Integration**: Super-resolution ($x4$) pre-processing to restore blurry or low-resolution images.
* **YOLOv8 Detection**: Custom-trained YOLOv8 (`best_v1.pt`,`best_v2.pt`) for accurate aircraft detection and model classification.
* **ResNet18 Logo Matching**: Fallback feature-extraction pipeline using Cosine Similarity to verify airline logos independently.
* **Offline-capable & Real-time UI**: Built-in Streamlit web app providing instant flight lookup and metadata aggregation.
* Dataset I used: https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft?resource=download (no need to download) 
---

## 🛠️ Project Structure

```text
Do-an-nganh/
├── logo/                   # Reference airline logo database
├── webapp/                 # Streamlit application source
│   ├── src/
│   │   ├── image_processor.py  # Real-ESRGAN handler
│   │   ├── metadata_worker.py  # Flight details processing
│   │   └── radar_api.py        # Radar / Flight lookup service
│   ├── app.py                  # Main Streamlit web application
│   └── RealESRGAN_x4plus.pth   # Pre-trained Real-ESRGAN weights
├── best_v1.pt              # Best trained version1 model weight
├── best_v2.pt              # Best trained version 2 model weight
├── requirements.txt        # Python dependencies
└── README.md
```
1. Installation & Setup
git clone [https://github.com/Toriki1111/Research-and-Development-of-Aircraft-Type-Detection-System-Combining-modified-YOLOv8-and-Real-ESRGAN.git](https://github.com/Toriki1111/Research-and-Development-of-Aircraft-Type-Detection-System-Combining-modified-YOLOv8-and-Real-ESRGAN.git)
cd Research-and-Development-of-Aircraft-Type-Detection-System-Combining-modified-YOLOv8-and-Real-ESRGAN

Intall PyTorch (do support CUDA / GPU)

Bash
```text
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
(If your com has only CPU change cu121 to cpu)
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

Frameworks & Libraries: PyTorch, Ultralytics YOLOv8, OpenCV, Pillow, Streamlit, gfpgan,git+https://github.com/xinntao/Real-ESRGAN.git

Usual error meet when install:
---
```text
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```
Fix
---
Access into: .venv\Lib\site-packages\basicsr\data\degradations.py (VScode,app bạn xài,...)
```text
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```
change to: 
```text
from torchvision.transforms.functional import rgb_to_grayscale (basically just delete _tensor and you are good to go)
```
Presentation in Vietnamese:
```text
https://canva.link/bow7fawxquqb7kp
```



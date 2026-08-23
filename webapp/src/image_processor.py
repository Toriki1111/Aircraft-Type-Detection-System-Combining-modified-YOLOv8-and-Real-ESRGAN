import os
import cv2
import numpy as np
from PIL import Image
import torch

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def init_realesrgan(model_name='RealESRGAN_x4plus.pth', tile=400, tile_pad=10, pre_pad=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Định vị đường dẫn tới thư mục webapp
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Thư mục webapp/
    model_path = os.path.join(base_dir, model_name)
    
    # 2. Nếu không tìm thấy trong webapp/, kiểm tra thư mục root của dự án
    if not os.path.exists(model_path):
        root_dir = os.path.dirname(base_dir)
        model_path = os.path.join(root_dir, model_name)

    print("--> ĐƯỜNG DẪN DÙNG LOAD WEIGHTS:", os.path.abspath(model_path) if not model_path.startswith('http') else model_path)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=True if device.type == 'cuda' else False,
        device=device
    )

    if hasattr(upsampler, 'model_path'):
        print("--> FILE WEIGHTS ĐÃ NẰM TẠI:", os.path.abspath(upsampler.model_path))
    return upsampler

# Khởi tạo upsampler toàn cục
try:
    realesrgan_upsampler = init_realesrgan()
    print("Loaded REAL-ESRGAN WEIGHTS")
except Exception as e:
    print(f"Chưa load được Real-ESRGAN weights: {e}")
    realesrgan_upsampler = None


def enhance_image_with_realesrgan(img_bgr, outscale=2):
    """
    Hàm làm nét ảnh bằng Real-ESRGAN phóng to x2
    """
    if realesrgan_upsampler is None:
        return img_bgr
    
    try:
        output_bgr, _ = realesrgan_upsampler.enhance(img_bgr, outscale=outscale)
        return output_bgr
    except Exception as e:
        print(f"Lỗi khi chạy Real-ESRGAN: {e}")
        return img_bgr


#PIPELINE
def preprocess_for_aircraft_detection(input_image, enable_esrgan=True):
    """
    Pipeline đơn giản & hiệu quả:
    - Chuyển đổi chuẩn hệ màu RGB/BGR giữa Streamlit và OpenCV.
    - Làm nét và tăng chi tiết bằng Real-ESRGAN.
    - Loại bỏ hoàn toàn bộ lọc Canny/Sky suppression gây nhiễu ảnh ban đêm.
    """
    # 1. Chuyển đổi đầu vào chuẩn BGR cho OpenCV & Real-ESRGAN
    if isinstance(input_image, Image.Image):
        img = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    elif isinstance(input_image, bytes):
        nparr = np.frombuffer(input_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(input_image, np.ndarray):
        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        else:
            img = input_image.copy()
    else:
        return None

    if img is None:
        return None

    # 2. Áp dụng Real-ESRGAN Super-Resolution
    if enable_esrgan and realesrgan_upsampler is not None:
        img = enhance_image_with_realesrgan(img, outscale=2)

    # 3. Trả về định dạng RGB cho Streamlit & YOLOv8
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"ESRGAN Status: {realesrgan_upsampler is not None}, Output Shape: {img_rgb.shape}")

    return img_rgb
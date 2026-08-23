from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_decimal_from_dms(dms, ref):
    """Chuyển đổi tọa độ từ dạng Độ-Phút-Giây (DMS) sang số Thập phân (Decimal)"""
    if not dms:
        return None
    degrees = float(dms[0])
    minutes = float(dms[1])
    seconds = float(dms[2])
    
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_gps_and_time(image_path):
    """Đọc file ảnh và trích xuất tọa độ GPS cùng thời gian chụp thực tế"""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        if not exif_data:
            return None, None, None
            
        readable_exif = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
        capture_time = readable_exif.get('DateTimeOriginal') or readable_exif.get('DateTime')
        
        gps_info = readable_exif.get('GPSInfo')
        if not gps_info:
            return None, None, capture_time
            
        gps_data = {GPSTAGS.get(tag, tag): gps_info[tag] for tag in gps_info}
        
        lat = get_decimal_from_dms(gps_data.get('GPSLatitude'), gps_data.get('GPSLatitudeRef'))
        lon = get_decimal_from_dms(gps_data.get('GPSLongitude'), gps_data.get('GPSLongitudeRef'))
        
        return lat, lon, capture_time
    except Exception:
        return None, None, None
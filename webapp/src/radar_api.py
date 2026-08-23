from FlightRadarAPI import FlightRadar24API
from FlightRadarAPI import Flight
import requests
import streamlit as st

def fetch_live_flights(lat, lon, radius_km=5):
    """Tìm các chuyến bay đang hoạt động quanh tọa độ GPS trong bán kính vùng quét"""
    try:
        fr_api = FlightRadar24API()
        lat_f = float(str(lat).replace(',', '.')) #lat vi do
        lon_f = float(str(lon).replace(',', '.')) #lon kinh do
        # 1. Lấy vùng quét từ thư viện
        bounds = fr_api.get_bounds_by_point(lat_f, lon_f, radius_km * 1000) #vi don vi goc trong thu vien yeu cau la m nen phai * 1000
        
        # 2. Tự cấu hình request thay vì gọi fr_api.get_flights()
        # Header này ép Server KHÔNG gửi gzip, chỉ gửi text thường
        headers = {
            "accept-encoding": "identity",  # Không nhận nén gzip
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" # giả lập trình duyệt để tránh rate limit 
        }
        
        url = f"https://data-cloud.flightradar24.com/zones/fcgi/feed.js?bounds={bounds}&faa=1&satellite=1&mlat=1&flarm=1&adsb=1&gnd=1&air=1&vehicles=1&estimated=1&maxage=14400&gliders=1&stats=1"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            flights = []
            
            # Ép dữ liệu thô dạng JSON về lại danh sách Object Flight giống hệt thư viện gốc
            for flight_id, flight_info in data.items():
                if flight_id in ["full_count", "version", "stats"]:
                    continue
                # Khởi tạo object máy bay để tương thích 100% với file app.py của bạn
                flights.append(Flight(flight_id, flight_info))
            return flights
        else:
            return []
    except Exception as e:
        st.error(f" Lỗi kết nối FlightRadar24 API: {e}")
        return []
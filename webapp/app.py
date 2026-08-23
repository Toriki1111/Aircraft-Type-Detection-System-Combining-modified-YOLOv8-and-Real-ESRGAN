import os
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
from streamlit_cropper import st_cropper

from src.image_processor import preprocess_for_aircraft_detection
from src.metadata_worker import extract_gps_and_time
from src.radar_api import fetch_live_flights

st.set_page_config(
    page_title="Aircraft Tracker AI", layout="wide", page_icon="✈️"
)

st.title(" ✈️ Aircraft Recognition & Radar Matching")
st.write("Hệ thống nhận diện máy bay tích hợp AI")
st.markdown("---")

# --- KHỞI TẠO BỘ NHỚ TẠM (SESSION STATE) ---
if "ai_results" not in st.session_state:
    st.session_state.ai_results = None
if "radar_flights" not in st.session_state:
    st.session_state.radar_flights = None
if "analyzed_image" not in st.session_state:
    st.session_state.analyzed_image = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "cropped_image_rgb" not in st.session_state:
    st.session_state.cropped_image_rgb = None

# --- SIDEBAR TÙY CHỌN ĐẦU VÀO ---
st.sidebar.header("⚙️ Tùy Chọn Đầu Vào")
app_mode = st.sidebar.radio(
    "Chọn phương thức:",
    [
        "💻 Upload ảnh từ thiết bị (PC/Lap/Phone)",
        "📱 Chụp ảnh trực tiếp (Camera Phone)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.header("Bộ Lọc Tiền Xử Lý Ảnh")
enable_preprocessing = st.sidebar.toggle("Real-ESRGAN & Edge Boost", value=False)

uploaded_file = None
if app_mode == "📱 Chụp ảnh trực tiếp (Camera Phone)":
    uploaded_file = st.sidebar.camera_input(
        "Screen Capture"
    )
else:
    uploaded_file = st.sidebar.file_uploader(
        "Chọn file ảnh máy bay từ máy tính của bạn",
        type=["jpg", "jpeg", "png"],
    )

# TỰ ĐỘNG REFRESH: Nếu đổi ảnh hoặc chụp ảnh mới, xóa bộ nhớ cũ ngay lập tức
if uploaded_file != st.session_state.last_uploaded_file:
    st.session_state.ai_results = None
    st.session_state.radar_flights = None
    st.session_state.analyzed_image = None
    st.session_state.cropped_image_rgb = None
    st.session_state.last_uploaded_file = uploaded_file

latitude, longitude, capture_time = None, None, None

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✂️ Cắt Vùng Ảnh Máy Bay")
        st.caption("💡 Kéo khung màu xanh để bao trọn chiếc máy bay.")
        raw_image = Image.open(uploaded_file).convert("RGB")

        # Key động giúp reset Cropper khi đổi ảnh
        cropper_key = f"cropper_{uploaded_file.name}_{raw_image.size[0]}x{raw_image.size[1]}"

        cropped_pil = st_cropper(
            raw_image,
            realtime_update=True,
            box_color="#00FF00",
            aspect_ratio=None,
            return_type="image",
            key=cropper_key,
        )
        st.session_state.cropped_image_rgb = np.array(cropped_pil)

    with col2:
        st.subheader("📍Vị Trí & Metadata")
        temp_path = "temp_user_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        latitude, longitude, capture_time = extract_gps_and_time(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if latitude and longitude:
            st.success("Thành công: Phát hiện dữ liệu GPS")
            st.text_input(
                "Vĩ độ (Latitude):", value=f"{latitude:.6f}", disabled=True
            )
            st.text_input(
                "Kinh độ (Longitude):", value=f"{longitude:.6f}", disabled=True
            )
            st.text_input(
                "Thời gian:",
                value=str(capture_time) if capture_time else "Không rõ",
                disabled=True,
            )
        else:
            st.warning("Không tìm thấy tọa độ GPS trong ảnh")
            latitude = st.number_input(
                "Nhập Vĩ độ (Latitude):", format="%.4f", value=10.8180
            )
            longitude = st.number_input(
                "Nhập Kinh độ (Longitude):", format="%.4f", value=106.6520
            )

        user_airport = (
            st.text_input("Sân bay theo dõi:", value="SGN").strip().upper()
        )

#NÚT KÍCH HOẠT PHÂN TÍCH
if st.button(
    "Kích Hoạt Phân Tích", type="primary", key="btn_run_analysis"
):
    if st.session_state.cropped_image_rgb is None:
        st.error("⚠️ Vui lòng chọn/tải ảnh lên trước khi kích hoạt!")
    else:
        with st.spinner("Đang quét đa kích thước (Multi-scale Scanning)..."):
            try:
                model = YOLO("webapp/best_v2.pt")
                raw_crop = st.session_state.cropped_image_rgb

                if enable_preprocessing:
                    processed_img = preprocess_for_aircraft_detection(raw_crop, enable_esrgan=True)
                    if processed_img is None:
                        processed_img = raw_crop
                else:
                    processed_img = raw_crop
                scales = [640, 704, 800]

                best_result = None
                max_conf = 0
                best_scale = 640

                for scale in scales:
                    results = model(
                        processed_img,
                        conf=0.4,
                        imgsz=scale,
                        iou=0.45, #iou bỏ các boundign box trùng lặp
                        agnostic_nms=True,
                        verbose=False, #Tránh việc 2 lớp khác nhau khoanh đè lên cùng một vật thể
                    )
                    if len(results[0].boxes) > 0:
                        current_max_conf = float(
                            results[0].boxes.conf.max().item()
                        )
                        if current_max_conf > max_conf:
                            max_conf = current_max_conf
                            best_result = results[0]
                            best_scale = scale

                if best_result is not None:
                    res_plotted_rgb = best_result.plot()
                    st.session_state.analyzed_image = res_plotted_rgb

                    parsed_ai_results = []
                    for box in best_result.boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = model.names[cls_id]
                        conf_val = float(box.conf[0].item())
                        parsed_ai_results.append(
                            {"name": cls_name, "conf": conf_val}
                        )
                    st.session_state.ai_results = parsed_ai_results

                    st.session_state.radar_flights = fetch_live_flights(
                        latitude, longitude
                    )

                    st.success(
                        f"🎯 Đã tìm thấy kết quả tốt nhất tại `imgsz={best_scale}` với Độ tin tưởng: **{max_conf:.2f}**"
                    )
                else:
                    st.session_state.analyzed_image = None
                    st.session_state.ai_results = []
                    st.session_state.radar_flights = fetch_live_flights(
                        latitude, longitude
                    )
                    st.warning(
                        "⚠️ Không phát hiện máy bay nào ngay cả khi đã quét qua tất cả kích thước."
                    )
            except Exception as e:
                st.error(f"Lỗi AI: {e}")

# --- HIỂN THỊ DỮ LIỆU KẾT QUẢ ---
if st.session_state.ai_results is not None:
    if st.session_state.analyzed_image is not None:
        st.image(
            st.session_state.analyzed_image,
            caption="Kết quả nhận diện từ AI",
            use_container_width=True,
        )

    has_ai_detection = len(st.session_state.ai_results) > 0

    if has_ai_detection:
        st.success(
            f"Found **{len(st.session_state.ai_results)}** planes in picture:"
        )
        for idx, item in enumerate(st.session_state.ai_results, 1):
            st.write(
                f"{idx}. ✈️ **{item['name']}** (Độ tin tưởng: {item['conf']*100:.1f}%)"
            )
    else:
        st.warning("⚠️ Không tìm thấy máy bay nào đạt độ tin tưởng tối thiểu.")

    # 2. KHU VỰC ĐỐI CHIẾU RADAR
    st.markdown("---")
    st.markdown("### 📊 Kết quả đối chiếu chéo với dữ liệu FlightRadar24")

    live_flights = st.session_state.radar_flights

    ai_predicted_brands = []
    if has_ai_detection:
        for item in st.session_state.ai_results:
            first_word = item["name"].split(" ")[0].upper()
            ai_predicted_brands.append(first_word)

    if live_flights:
        st.info(
            f"Đang truy xuất dữ liệu không lưu quanh tọa độ sân bay: **{user_airport}**"
        )

        filter_option = st.radio(
            "Chuyến bay hiển thị:",
            options=[
                " Ưu tiên theo AI nhận diện",
                " Tất cả chuyến bay trong khu vực",
            ],
            horizontal=True,
            key="radar_filter_radio",
        )

        departures = []
        arrivals = []

        is_filter_by_ai = "Ưu tiên theo AI" in filter_option

        for fl in live_flights:
            r_aircraft = str(
                getattr(fl, "aircraft_code", "")
                or getattr(fl, "aircraft_model", "N/A")
            ).upper()
            r_airline = getattr(
                fl,
                "airline_name",
                getattr(fl, "airline_icao", "Hãng bay không xác định"),
            )
            r_number = getattr(fl, "number", "N/A")
            r_reg = getattr(fl, "registration", "N/A")
            r_alt = getattr(fl, "altitude", "N/A")
            r_speed = getattr(fl, "ground_speed", "N/A")
            r_heading = getattr(fl, "heading", "N/A")

            r_origin = str(
                getattr(
                    fl,
                    "origin_airport_iata",
                    getattr(fl, "origin_airport_icao", "N/A"),
                )
            ).upper()
            r_dest = str(
                getattr(
                    fl,
                    "destination_airport_iata",
                    getattr(fl, "destination_airport_icao", "N/A"),
                )
            ).upper()

            is_match = False

            if not is_filter_by_ai:
                # Chọn "Tất cả chuyến bay trong khu vực" -> Hiện hết
                is_match = True
            elif is_filter_by_ai and has_ai_detection:
                # Lọc theo AI khi AI thực sự tìm thấy máy bay
                for target_brand in ai_predicted_brands:
                    if target_brand == "AIRBUS" and r_aircraft.startswith("A"):
                        is_match = True
                        break
                    elif target_brand == "BOEING" and r_aircraft.startswith("B"):
                        is_match = True
                        break
                    elif (
                        target_brand in r_aircraft
                        or r_aircraft in target_brand
                    ):
                        is_match = True
                        break

            if is_match:
                flight_data = {
                    "number": r_number,
                    "aircraft": r_aircraft,
                    "airline": r_airline,
                    "reg": r_reg,
                    "origin": r_origin,
                    "dest": r_dest,
                    "alt": r_alt,
                    "speed": r_speed,
                    "heading": r_heading,
                }
                if user_airport in r_origin:
                    departures.append(flight_data)
                else:
                    arrivals.append(flight_data)

        total_matches = len(departures) + len(arrivals)

        # XỬ LÝ THÔNG BÁO VÀ RENDER LOGIC CHÍNH XÁC
        if is_filter_by_ai and not has_ai_detection:
            st.warning(
                "⚠️ AI không tìm thấy máy bay nào trong ảnh nên không thể lọc theo nhận diện. "
                "Vui lòng chọn **'Tất cả chuyến bay trong khu vực'** để xem dữ liệu Radar."
            )
        elif total_matches > 0:
            if not is_filter_by_ai:
                st.success(
                    f"Đang hiển thị **Toàn bộ {total_matches} chuyến bay** đang hoạt động trong bán kính radar."
                )
            else:
                st.success(
                    f"Tìm thấy **{total_matches} chuyến bay** phù hợp với nhận diện của AI."
                )

            tab_arrival, tab_departure = st.tabs(
                [
                    f"🛬 Chuyến Bay Đến / Arrived ({len(arrivals)})",
                    f"🛫 Chuyến Bay Đi / Departure ({len(departures)})",
                ]
            )

            def render_flight_list(flight_list, tab_object):
                with tab_object:
                    if not flight_list:
                        st.info(
                            f"Không có chuyến bay nào tương ứng với mã sân bay `{user_airport}`."
                        )
                        return
                    for f in flight_list:
                        with st.expander(
                            f"✈️ Radar Live: Chuyến bay {f['number']} ({f['aircraft']}) | {f['airline']}"
                        ):
                            col_b1, col_b2, col_b3 = st.columns(3)
                            with col_b1:
                                st.markdown("**📋 Basic Information**")
                                st.write(
                                    f"- **Hãng khai thác / Brand:** {f['airline']}"
                                )
                                st.write(
                                    f"- **Số hiệu chuyến /No:** {f['number']}"
                                )
                                st.write(
                                    f"- **Mã đăng ký (Reg):** `{f['reg']}`"
                                )
                            with col_b2:
                                st.markdown("**🗺️ Flight Route**")
                                st.write(
                                    f"- **Sân bay đi / Departure:** `{f['origin']}`"
                                )
                                st.write(
                                    f"- **Sân bay đến / Arrived:** `{f['dest']}`"
                                )
                                st.write(
                                    f"- **Dòng máy bay:** `{f['aircraft']}`"
                                )
                            with col_b3:
                                st.markdown("**⚡ Aircraft Status**")
                                alt_val = (
                                    float(f["alt"])
                                    if str(f["alt"])
                                    .replace(".", "", 1)
                                    .isdigit()
                                    else 0
                                )
                                spd_val = (
                                    float(f["speed"])
                                    if str(f["speed"])
                                    .replace(".", "", 1)
                                    .isdigit()
                                    else 0
                                )
                                st.write(
                                    f"- **Độ cao / Altitude:** {f['alt']} ft / {alt_val * 0.3048:.2f} m"
                                )
                                st.write(
                                    f"- **Tốc độ / Speed:** {f['speed']} knots / {spd_val * 1.852:.2f} km/h"
                                )
                                st.write(
                                    f"- **Hướng bay / Direction:** {f['heading']}°"
                                )

            render_flight_list(arrivals, tab_arrival)
            render_flight_list(departures, tab_departure)
        else:
            st.warning(
                f"Không tìm thấy chuyến bay nào khớp bộ lọc xung quanh sân bay `{user_airport}`. Hãy thử chuyển sang bộ lọc 'Tất cả chuyến bay'."
            )
    else:
        st.warning(
            "Không thể lấy dữ liệu không lưu từ FlightRadar24 xung quanh tọa độ hiện tại."
        )
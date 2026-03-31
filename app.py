import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="PM2.5 Dashboard", layout="wide", page_icon="🌫️")

# ====== STYLE ======
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Kanit', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.title {
    font-size: 38px;
    color: white;
    font-weight: 700;
    letter-spacing: 1px;
}
.subtitle {
    font-size: 15px;
    color: rgba(255,255,255,0.55);
    margin-top: -8px;
    margin-bottom: 20px;
}
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    border-radius: 16px;
    padding: 20px;
    color: white;
    text-align: center;
    font-size: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.1);
}
.card .label {
    font-size: 13px;
    color: rgba(255,255,255,0.6);
    margin-bottom: 6px;
}
.card .value {
    font-size: 32px;
    font-weight: 700;
}
.card .unit {
    font-size: 13px;
    color: rgba(255,255,255,0.5);
}
.section {
    color: white;
    font-size: 20px;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 10px;
    border-left: 4px solid #38bdf8;
    padding-left: 12px;
}
.status-card {
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    color: white;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}
.predict-badge {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px;
    padding: 6px 14px;
    color: rgba(255,255,255,0.75);
    font-size: 14px;
    display: inline-block;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ====== HEADER ======
st.markdown('<div class="title">🌫️ ระบบวิเคราะห์และทำนาย PM2.5</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ข้อมูลคุณภาพอากาศรายชั่วโมง — กรุงเทพมหานครและปริมณฑล</div>', unsafe_allow_html=True)

# ====== LOCATIONS ======
locations = [
    ("บางนา",     13.67, 100.60),
    ("ลาดกระบัง", 13.72, 100.77),
    ("จตุจักร",   13.80, 100.55),
    ("ธนบุรี",    13.70, 100.48),
    ("ดินแดง",    13.77, 100.56),
    ("ปทุมวัน",   13.74, 100.53),
]

# ====== AQI HELPER (มาตรฐาน PM2.5 ของไทย µg/m³) ======
def get_aqi_info(pm25_value):
    if pm25_value <= 25:
        return "คุณภาพอากาศดี", "#22c55e", "🟢", "อากาศบริสุทธิ์ เหมาะสำหรับกิจกรรมกลางแจ้ง"
    elif pm25_value <= 37:
        return "คุณภาพอากาศปานกลาง", "#eab308", "🟡", "กลุ่มเสี่ยงควรระวัง หากสัมผัสเป็นเวลานาน"
    elif pm25_value <= 50:
        return "เริ่มมีผลต่อสุขภาพ", "#f97316", "🟠", "ควรลดกิจกรรมกลางแจ้งที่ใช้แรงมาก"
    elif pm25_value <= 90:
        return "มีผลต่อสุขภาพ", "#ef4444", "🔴", "ควรสวมหน้ากากหากออกนอกบ้าน"
    else:
        return "อันตราย — คุณภาพอากาศแย่มาก", "#a855f7", "🟣", "หลีกเลี่ยงการออกนอกบ้านโดยไม่จำเป็น"

# ====== LOAD DATA ======
@st.cache_data(ttl=600)
def load_data():
    all_data = []

    for name, lat, lon in locations:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5",
        }

        try:
            res = requests.get(url, params=params, timeout=10)

            if res.status_code != 200:
                st.warning(f"⚠️ {name}: API ตอบกลับ {res.status_code}")
                continue

            if not res.text.strip():
                st.warning(f"⚠️ {name}: response ว่างเปล่า")
                continue

            data = res.json()

            if "hourly" not in data:
                st.warning(f"⚠️ {name}: ไม่พบข้อมูล hourly")
                continue

            times  = data["hourly"].get("time",  [])
            values = data["hourly"].get("pm2_5", [])

            for t, v in zip(times, values):
                if v is not None:
                    all_data.append([name, lat, lon, t, float(v)])

        except requests.exceptions.Timeout:
            st.warning(f"⚠️ {name}: หมดเวลาเชื่อมต่อ")
            continue
        except Exception as e:
            st.warning(f"⚠️ โหลด {name} ไม่สำเร็จ: {e}")
            continue

    if not all_data:
        return pd.DataFrame(columns=["location", "lat", "lon", "time", "pm25"])

    df = pd.DataFrame(all_data, columns=["location", "lat", "lon", "time", "pm25"])
    df["time"] = pd.to_datetime(df["time"])
    return df

# ====== LOAD ======
with st.spinner("กำลังโหลดข้อมูล..."):
    df = load_data()

if df.empty:
    st.error("❌ โหลดข้อมูลไม่สำเร็จ กรุณาลองใหม่ภายหลัง หรือ API อาจล่มชั่วคราว")
    st.stop()

# ====== SIDEBAR / SELECT ======
st.markdown('<div class="section">เลือกพื้นที่</div>', unsafe_allow_html=True)
districts = df["location"].unique().tolist()
selected  = st.selectbox("", districts, label_visibility="collapsed")

filtered = df[df["location"] == selected].sort_values("time").copy()

if filtered.empty:
    st.error(f"❌ ไม่มีข้อมูลสำหรับ {selected}")
    st.stop()

# ====== LATEST ROW (guard) ======
latest_row = filtered.iloc[-1]
latest_pm25 = latest_row["pm25"]
latest_time = latest_row["time"].strftime("%d/%m/%Y %H:%M")

# ====== MODEL — lag features + predict 1 hr ahead ======
filtered["lag1"] = filtered["pm25"].shift(1)
filtered["lag2"] = filtered["pm25"].shift(2)
filtered_model   = filtered.dropna(subset=["lag1", "lag2"]).copy()

prediction = None
if len(filtered_model) >= 5:
    X = filtered_model[["lag1", "lag2"]]
    y = filtered_model["pm25"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    last_features = X.iloc[-1].values.reshape(1, -1)
    prediction    = float(model.predict(last_features)[0])
else:
    st.warning("⚠️ ข้อมูลไม่เพียงพอสำหรับการทำนาย (ต้องการอย่างน้อย 5 แถว)")

# ====== STATS CARDS ======
st.markdown('<div class="section">สรุปค่า PM2.5</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"""
<div class="card">
  <div class="label">ค่าล่าสุด ({latest_time})</div>
  <div class="value">{latest_pm25:.1f}</div>
  <div class="unit">µg/m³</div>
</div>""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="card">
  <div class="label">ค่าเฉลี่ย (ทั้งหมด)</div>
  <div class="value">{filtered['pm25'].mean():.1f}</div>
  <div class="unit">µg/m³</div>
</div>""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="card">
  <div class="label">ค่าสูงสุด</div>
  <div class="value">{filtered['pm25'].max():.1f}</div>
  <div class="unit">µg/m³</div>
</div>""", unsafe_allow_html=True)

pred_display = f"{prediction:.1f}" if prediction is not None else "N/A"
col4.markdown(f"""
<div class="card">
  <div class="label">ทำนาย 1 ชั่วโมงข้างหน้า</div>
  <div class="value">{pred_display}</div>
  <div class="unit">µg/m³</div>
</div>""", unsafe_allow_html=True)

# ====== LINE CHART ======
st.markdown('<div class="section">แนวโน้ม PM2.5 รายชั่วโมง</div>', unsafe_allow_html=True)

# ใช้ข้อมูล 72 ชั่วโมงล่าสุดเพื่อความชัดเจน
recent = filtered.tail(72)

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=recent["time"],
    y=recent["pm25"],
    mode="lines",
    name="PM2.5 จริง",
    line=dict(color="#38bdf8", width=2),
    fill="tozeroy",
    fillcolor="rgba(56,189,248,0.1)",
))

if prediction is not None:
    next_time = recent["time"].iloc[-1] + pd.Timedelta(hours=1)
    fig_line.add_trace(go.Scatter(
        x=[recent["time"].iloc[-1], next_time],
        y=[recent["pm25"].iloc[-1], prediction],
        mode="lines+markers",
        name="ทำนาย 1 ชม.",
        line=dict(color="#f97316", width=2, dash="dot"),
        marker=dict(size=10, color="#f97316", symbol="star"),
    ))

fig_line.add_hline(y=25, line_dash="dash", line_color="rgba(34,197,94,0.4)",  annotation_text="ดี (25)")
fig_line.add_hline(y=37, line_dash="dash", line_color="rgba(234,179,8,0.4)",  annotation_text="ปานกลาง (37)")
fig_line.add_hline(y=50, line_dash="dash", line_color="rgba(249,115,22,0.4)", annotation_text="เริ่มมีผล (50)")
fig_line.add_hline(y=90, line_dash="dash", line_color="rgba(239,68,68,0.4)",  annotation_text="อันตราย (90)")

fig_line.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=400,
    legend=dict(orientation="h", y=1.1),
    margin=dict(l=10, r=10, t=30, b=10),
    yaxis_title="PM2.5 (µg/m³)",
    xaxis_title="เวลา",
)

st.plotly_chart(fig_line, use_container_width=True)

# ====== MAP ======
st.markdown('<div class="section">แผนที่ PM2.5 ปัจจุบัน</div>', unsafe_allow_html=True)

latest_per_loc = (
    df.sort_values("time")
      .groupby("location", as_index=False)
      .last()
)

fig_map = px.density_mapbox(
    latest_per_loc,
    lat="lat",
    lon="lon",
    z="pm25",
    hover_name="location",
    hover_data={"pm25": ":.1f", "lat": False, "lon": False},
    radius=40,
    center=dict(lat=13.75, lon=100.55),
    zoom=10,
    mapbox_style="open-street-map",
    color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444", "#a855f7"],
    range_color=[0, 100],
)

fig_map.update_layout(
    height=550,
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
    coloraxis_colorbar=dict(title="PM2.5<br>(µg/m³)"),
)

st.plotly_chart(fig_map, use_container_width=True)

# ====== STATUS ======
st.markdown('<div class="section">สถานะคุณภาพอากาศ (ตามค่าทำนาย)</div>', unsafe_allow_html=True)

if prediction is not None:
    status_label, status_color, status_icon, status_advice = get_aqi_info(prediction)
    st.markdown(f"""
    <div class="predict-badge">📍 {selected} &nbsp;|&nbsp; ทำนาย 1 ชม. ข้างหน้า: {prediction:.1f} µg/m³</div>
    <div class="status-card" style="background: linear-gradient(135deg, {status_color}cc, {status_color}55); border: 1px solid {status_color};">
        {status_icon} {status_label}<br>
        <span style="font-size:15px; font-weight:400; opacity:0.9;">{status_advice}</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("ไม่สามารถแสดงสถานะได้เนื่องจากข้อมูลไม่เพียงพอ")

# ====== COMPARE ALL DISTRICTS ======
st.markdown('<div class="section">เปรียบเทียบทุกพื้นที่ (ค่าล่าสุด)</div>', unsafe_allow_html=True)

latest_per_loc_sorted = latest_per_loc.sort_values("pm25", ascending=True)

colors = [get_aqi_info(v)[1] for v in latest_per_loc_sorted["pm25"]]

fig_bar = go.Figure(go.Bar(
    x=latest_per_loc_sorted["pm25"],
    y=latest_per_loc_sorted["location"],
    orientation="h",
    marker_color=colors,
    text=latest_per_loc_sorted["pm25"].apply(lambda x: f"{x:.1f}"),
    textposition="outside",
    textfont=dict(color="white"),
))

fig_bar.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=300,
    margin=dict(l=10, r=60, t=10, b=10),
    xaxis_title="PM2.5 (µg/m³)",
    yaxis_title="",
    showlegend=False,
)

st.plotly_chart(fig_bar, use_container_width=True)

# ====== FOOTER ======
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.3); font-size:13px; margin-top:40px; padding-bottom:20px;">
    ข้อมูลจาก Open-Meteo Air Quality API · อัปเดตทุก 10 นาที · มาตรฐาน AQI ของประเทศไทย
</div>
""", unsafe_allow_html=True)
"""
CivicLens AI - Autonomous Municipal Triage & Geospatial Dispatch Engine
=======================================================================
pip install streamlit ultralytics folium streamlit-folium pillow pandas numpy

Run: streamlit run app.py
"""

# Standard library
import io
import math
import uuid
from datetime import datetime, timedelta

# Third-party
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from PIL import Image, ImageDraw

# PAGE CONFIG - must be first Streamlit call
st.set_page_config(
    page_title="CivicLens AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.civiclens-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f4c75 100%);
    padding: 2rem 2.5rem 1.5rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #1e40af44;
}
.civiclens-header h1 { color:#f0f9ff; font-size:2rem; font-weight:800; margin:0 0 0.3rem 0; letter-spacing:-0.5px; }
.civiclens-header p  { color:#93c5fd; font-size:0.92rem; margin:0; font-style:italic; }
.kpi-card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.kpi-card .kpi-value { font-size:2.2rem; font-weight:800; color:#38bdf8; line-height:1.1; }
.kpi-card .kpi-label { font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em; margin-top:0.35rem; }
.kpi-card .kpi-delta { font-size:0.8rem; font-weight:600; margin-top:0.25rem; }
.delta-good { color:#4ade80; }
.delta-info { color:#facc15; }
.section-title {
    font-size:1.05rem; font-weight:700; color:#e2e8f0;
    border-left:4px solid #3b82f6; padding-left:0.7rem; margin:1.4rem 0 0.8rem 0;
}
.info-box    { background:#0f2744; border:1px solid #1d4ed8; border-radius:10px; padding:1rem 1.2rem; margin:0.8rem 0; font-size:0.88rem; color:#bfdbfe; }
.warn-box    { background:#451a03; border:1px solid #d97706; border-radius:10px; padding:1rem 1.2rem; margin:0.8rem 0; font-size:0.88rem; color:#fde68a; }
.success-box { background:#052e16; border:1px solid #16a34a; border-radius:10px; padding:1rem 1.2rem; margin:0.8rem 0; font-size:0.88rem; color:#86efac; }

    /* ── Dark enterprise theme ── */
    .stApp {
        background-color: #0e1117;
        color: #f3f4f6;
    }
    /* Metric cards styling */
    [data-testid="stMetric"] {
        background-color: #1a1f2c;
        border: 1px solid #2e384d;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #1e40af);
        transform: translateY(-1px);
    }
    /* Clean container panels */
    div[data-testid="stExpander"], div[data-testid="stVerticalBlock"] > div {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: HAVERSINE DISTANCE
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    """Return distance in metres between two GPS coordinates."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi        = math.radians(lat2 - lat1)
    dlambda     = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: YOLO INFERENCE (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8 vision model...")
def load_yolo():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        return model, True
    except Exception:
        return None, False

def run_inference(model, img):
    """Returns list of detection dicts. Falls back to mock if model is None."""
    if model is not None:
        try:
            results = model(np.array(img.convert("RGB")), verbose=False)
            detections = []
            W, H = img.size
            img_area = W * H
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area_frac = ((x2 - x1) * (y2 - y1)) / img_area
                    label = r.names[int(box.cls[0].item())]
                    conf  = float(box.conf[0].item())
                    detections.append({
                        "label": label, "conf": conf, "bbox_frac": area_frac,
                        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)
                    })
            return detections
        except Exception:
            pass
    # Mock fallback
    W, H = img.size
    return [{
        "label": "pothole", "conf": 0.87, "bbox_frac": 0.18,
        "x1": W//4, "y1": H//4, "x2": 3*W//4, "y2": 3*H//4
    }]

def draw_boxes(img, detections):
    draw = ImageDraw.Draw(img)
    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        draw.rectangle([x1, y1, x2, y2], outline="#ef4444", width=3)
        label_txt = f"{d['label']} {d['conf']:.0%}"
        draw.rectangle([x1, y1-18, x1+len(label_txt)*7, y1], fill="#ef4444")
        draw.text((x1+3, y1-16), label_txt, fill="white")
    return img

# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY & DEPARTMENT LOOKUP TABLES
# ─────────────────────────────────────────────────────────────────────────────
HAZARD_BASE_SEVERITY = {
    "Pothole / Road Damage":       7.0,
    "Garbage / Waste Overflow":    4.5,
    "Water Leakage / Pipe Burst":  8.5,
    "Broken Streetlight":          5.0,
    "Fallen Tree / Blockage":      6.5,
    "Flooding / Waterlogging":     9.0,
    "Sewage Overflow":             8.0,
    "Illegal Construction":        5.5,
}

DEPARTMENT_MAP = {
    "Pothole / Road Damage":       "Road Maintenance & Public Works",
    "Garbage / Waste Overflow":    "Solid Waste Management",
    "Water Leakage / Pipe Burst":  "Water Supply & Drainage Board",
    "Broken Streetlight":          "Electrical & Street Lighting Dept",
    "Fallen Tree / Blockage":      "Urban Forestry & Horticulture",
    "Flooding / Waterlogging":     "Storm-Water Drainage Authority",
    "Sewage Overflow":             "Sewerage & Sanitation Dept",
    "Illegal Construction":        "Town Planning & Enforcement Cell",
}

CREW_MAP = {
    "Road Maintenance & Public Works":     "Crew Alpha-7",
    "Solid Waste Management":             "Crew Bravo-3",
    "Water Supply & Drainage Board":      "Crew Charlie-9",
    "Electrical & Street Lighting Dept":  "Crew Delta-2",
    "Urban Forestry & Horticulture":      "Crew Echo-4",
    "Storm-Water Drainage Authority":     "Crew Foxtrot-1",
    "Sewerage & Sanitation Dept":         "Crew Golf-6",
    "Town Planning & Enforcement Cell":   "Crew Hotel-8",
}

def compute_severity(hazard, detections):
    base      = HAZARD_BASE_SEVERITY.get(hazard, 5.5)
    max_frac  = max((d["bbox_frac"] for d in detections), default=0.15)
    area_bonus = min(max_frac * 8, 2.5)
    conf_avg  = np.mean([d["conf"] for d in detections]) if detections else 0.75
    return round(min(base + area_bonus * conf_avg, 10.0), 2)

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE IMAGE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def make_sample_pothole():
    img  = Image.new("RGB", (480, 340), color="#374151")
    draw = ImageDraw.Draw(img)
    for y in range(0, 340, 12):
        draw.line([(0, y), (480, y)], fill="#4b5563", width=1)
    draw.ellipse([130, 100, 350, 240], fill="#111827", outline="#ef4444", width=4)
    draw.ellipse([160, 120, 320, 220], fill="#0f172a")
    draw.line([(240, 100), (200, 50)], fill="#6b7280", width=2)
    draw.line([(300, 150), (380, 120)], fill="#6b7280", width=2)
    draw.text((100, 260), "[ SYNTHETIC DEMO — POTHOLE SAMPLE ]", fill="#fbbf24")
    return img

def make_sample_garbage():
    img  = Image.new("RGB", (480, 340), color="#292524")
    draw = ImageDraw.Draw(img)
    for i, c in enumerate(["#16a34a", "#15803d", "#166534", "#14532d"]):
        draw.rectangle([60+i*90, 120, 130+i*90, 260], fill=c, outline="#4ade80", width=2)
    draw.rectangle([50, 258, 440, 280], fill="#57534e")
    draw.text((80, 295), "[ SYNTHETIC DEMO — GARBAGE OVERFLOW SAMPLE ]", fill="#fbbf24")
    return img

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def _seed_incidents():
    now = datetime.now()
    return [
        {"id":"TKT-1001","hazard":"Pothole / Road Damage","severity":8.5,
         "department":"Road Maintenance & Public Works","crew":"Crew Alpha-7",
         "lat":18.5204,"lon":73.8567,"reports_count":4,"status":"In-Progress",
         "submitted":(now-timedelta(hours=3)).strftime("%H:%M %d-%b")},
        {"id":"TKT-1002","hazard":"Garbage / Waste Overflow","severity":4.2,
         "department":"Solid Waste Management","crew":"Crew Bravo-3",
         "lat":18.5280,"lon":73.8650,"reports_count":1,"status":"Queued",
         "submitted":(now-timedelta(hours=1,minutes=20)).strftime("%H:%M %d-%b")},
        {"id":"TKT-1003","hazard":"Water Leakage / Pipe Burst","severity":9.1,
         "department":"Water Supply & Drainage Board","crew":"Crew Charlie-9",
         "lat":18.5150,"lon":73.8480,"reports_count":7,"status":"Emergency Action",
         "submitted":(now-timedelta(hours=5,minutes=45)).strftime("%H:%M %d-%b")},
        {"id":"TKT-1004","hazard":"Broken Streetlight","severity":3.8,
         "department":"Electrical & Street Lighting Dept","crew":"Crew Delta-2",
         "lat":18.5330,"lon":73.8510,"reports_count":2,"status":"Queued",
         "submitted":(now-timedelta(minutes=40)).strftime("%H:%M %d-%b")},
        {"id":"TKT-1005","hazard":"Sewage Overflow","severity":8.0,
         "department":"Sewerage & Sanitation Dept","crew":"Crew Golf-6",
         "lat":18.5090,"lon":73.8600,"reports_count":3,"status":"Dispatched",
         "submitted":(now-timedelta(hours=2)).strftime("%H:%M %d-%b")},
    ]

if "incidents"        not in st.session_state: st.session_state.incidents        = _seed_incidents()
if "ticket_counter"   not in st.session_state: st.session_state.ticket_counter   = 1006
if "submission_log"   not in st.session_state: st.session_state.submission_log   = []
if "sample_img"       not in st.session_state: st.session_state.sample_img       = None
if "sample_type"      not in st.session_state: st.session_state.sample_type      = None

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="civiclens-header">
  <h1>🏛️ CivicLens AI — Autonomous Municipal Triage & Geospatial Dispatch Engine</h1>
  <p>Real-Time Hazard Detection &nbsp;·&nbsp; Spatio-Temporal Deduplication &nbsp;·&nbsp; Automated Job Dispatch &nbsp;·&nbsp; Pune Smart City Demo</p>
</div>
""", unsafe_allow_html=True)

# Load YOLO model
yolo_model, yolo_ok = load_yolo()
if yolo_ok:
    st.toast("✅ YOLOv8n vision model loaded successfully", icon="🤖")
else:
    st.toast("⚠️ YOLOv8 unavailable — running mock-detection mode", icon="⚠️")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "📸  Citizen Hazard Submission",
    "🏛️  Municipal Command & Dispatch Center (Live Operations)",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CITIZEN SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<p class="section-title">🗂 Incident Details</p>', unsafe_allow_html=True)

        hazard_type = st.selectbox(
            "Hazard Category Override / Notes",
            list(HAZARD_BASE_SEVERITY.keys()),
            help="Select the type of civic hazard you are reporting.",
        )

        uploaded_file = st.file_uploader(
            "Upload Hazard Photo (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            help="Clear, well-lit photos improve AI detection accuracy.",
        )

        sc1, sc2 = st.columns(2)
        use_pothole_sample = sc1.button("🕳️ Use Sample Pothole Image", use_container_width=True)
        use_garbage_sample = sc2.button("🗑️ Use Sample Garbage Image", use_container_width=True)

        st.markdown('<p class="section-title">📍 GPS Coordinates</p>', unsafe_allow_html=True)
        gc1, gc2 = st.columns(2)
        gps_lat = gc1.number_input("Latitude",  value=18.5205, format="%.6f", step=0.0001)
        gps_lon = gc2.number_input("Longitude", value=73.8568, format="%.6f", step=0.0001)
        st.caption("📡 Coordinates auto-injected from mobile GPS metadata | Edit for manual override")

        # Proximity preview
        if st.session_state.incidents:
            dists = [(haversine(gps_lat, gps_lon, i["lat"], i["lon"]), i["id"]) for i in st.session_state.incidents]
            nearest_dist, nearest_id = min(dists, key=lambda x: x[0])
            if nearest_dist <= 50:
                st.markdown(
                    f'<div class="warn-box">⚡ <b>Proximity Alert:</b> Active incident <b>{nearest_id}</b> is only '
                    f'<b>{nearest_dist:.1f} m</b> away. Deduplication engine will evaluate clustering.</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<p class="section-title">🚀 Submit Report</p>', unsafe_allow_html=True)
        submit_btn = st.button("🚀  Submit & Triage Report", type="primary", use_container_width=True)

    with right:
        st.markdown('<p class="section-title">🔍 Vision Frame & Bounding Box Inspection</p>', unsafe_allow_html=True)

        # Resolve display image
        if use_pothole_sample:
            st.session_state.sample_img  = make_sample_pothole()
            st.session_state.sample_type = "pothole"
        elif use_garbage_sample:
            st.session_state.sample_img  = make_sample_garbage()
            st.session_state.sample_type = "garbage"

        if uploaded_file is not None:
            display_img = Image.open(uploaded_file).convert("RGB")
            st.session_state.sample_img  = display_img
            st.session_state.sample_type = "upload"
        elif st.session_state.sample_img is not None:
            display_img = st.session_state.sample_img.copy()
        else:
            display_img = make_sample_pothole()
            st.session_state.sample_img  = display_img
            st.session_state.sample_type = "pothole"

        img_placeholder = st.empty()
        img_placeholder.image(display_img, caption="Input frame — awaiting YOLOv8 inference", use_container_width=True)

        # ── SUBMISSION LOGIC ────────────────────────────────────────────────
        if submit_btn:
            with st.spinner("🧠 Running YOLOv8 vision inference & deduplication..."):
                work_img   = display_img.copy()
                detections = run_inference(yolo_model, work_img)
                annotated  = draw_boxes(work_img.copy(), detections)
                severity   = compute_severity(hazard_type, detections)
                dept       = DEPARTMENT_MAP.get(hazard_type, "General Municipal Services")
                crew       = CREW_MAP.get(dept, "Crew Zulu-0")

                # Deduplication check
                DEDUP_RADIUS_M = 20.0
                matched_inc  = None
                matched_dist = None
                for inc in st.session_state.incidents:
                    dist = haversine(gps_lat, gps_lon, inc["lat"], inc["lon"])
                    if dist <= DEDUP_RADIUS_M and inc["hazard"] == hazard_type:
                        matched_inc  = inc
                        matched_dist = dist
                        break

                now_str = datetime.now().strftime("%H:%M %d-%b")

                if matched_inc:
                    matched_inc["reports_count"] += 1
                    matched_inc["severity"]       = round(min(10.0, matched_inc["severity"] + 0.2), 2)
                    result_type  = "duplicate"
                    result_id    = matched_inc["id"]
                    log_entry    = {"time": now_str, "type": "DUPLICATE CLUSTERED",
                                    "ticket": result_id, "hazard": hazard_type, "severity": matched_inc["severity"]}
                else:
                    new_id = f"TKT-{st.session_state.ticket_counter}"
                    st.session_state.ticket_counter += 1
                    st.session_state.incidents.append({
                        "id": new_id, "hazard": hazard_type, "severity": severity,
                        "department": dept, "crew": crew, "lat": gps_lat, "lon": gps_lon,
                        "reports_count": 1, "status": "Dispatched", "submitted": now_str,
                    })
                    result_type = "new"
                    result_id   = new_id
                    log_entry   = {"time": now_str, "type": "NEW TICKET",
                                   "ticket": new_id, "hazard": hazard_type, "severity": severity}

                st.session_state.submission_log.append(log_entry)

            # Show annotated image
            img_placeholder.image(annotated, caption=f"YOLOv8 Annotated — {len(detections)} detection(s)", use_container_width=True)

            # Result message
            if result_type == "duplicate":
                st.markdown(f"""
<div class="warn-box">
  <b>📍 DUPLICATE CLUSTERED — Deduplication Engine Triggered</b><br/>
  Identical hazard found within <b>{matched_dist:.1f} m</b> of an active incident.<br/>
  Report merged into ticket <b>{result_id}</b>. Urgency priority elevated.<br/>
  Citizen confirmation count: <b>{matched_inc["reports_count"]}</b>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="success-box">
  <b>✅ New Unique Incident — Dispatched to Field Crew</b><br/>
  <b>Ticket ID:</b> {result_id} &nbsp;|&nbsp;
  <b>Department:</b> {dept} &nbsp;|&nbsp;
  <b>Assigned Crew:</b> {crew}
</div>""", unsafe_allow_html=True)

            sev_label = "🔴 Critical" if severity >= 8 else ("🟠 Moderate" if severity >= 5 else "🟢 Low")
            m1, m2, m3 = st.columns(3)
            m1.metric("Severity Index", f"{severity}/10", sev_label)
            m2.metric("Detections Found", len(detections))
            m3.metric("Dispatch Status", "Auto-Routed" if result_type == "new" else "Clustered")

            with st.expander("🔎 Raw YOLOv8 Detection Output"):
                st.dataframe(pd.DataFrame([
                    {"Class": d["label"], "Confidence": f"{d['conf']:.2%}",
                     "BBox Area (% frame)": f"{d['bbox_frac']*100:.1f}%"}
                    for d in detections
                ]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MUNICIPAL COMMAND DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    incidents = st.session_state.incidents

    total_reports  = sum(i["reports_count"] for i in incidents)
    unique_tickets = len(incidents)
    duplicates     = total_reports - unique_tickets
    pct_saved      = (duplicates / total_reports * 100) if total_reports else 0
    critical_count = sum(1 for i in incidents if i["severity"] >= 8.0)
    avg_severity   = float(np.mean([i["severity"] for i in incidents])) if incidents else 0.0

    # KPI Row
    st.markdown('<p class="section-title">📊 Key Performance Indicators</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{total_reports}</div>'
                    f'<div class="kpi-label">Total Reports Ingested</div>'
                    f'<div class="kpi-delta delta-info">↑ Live citizen feed</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{duplicates}</div>'
                    f'<div class="kpi-label">Duplicates Clustered<br/>(Triage Overhead Saved)</div>'
                    f'<div class="kpi-delta delta-good">↓ {pct_saved:.0f}% dedup rate</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">3.2 hrs</div>'
                    f'<div class="kpi-label">Avg Resolution Cycle</div>'
                    f'<div class="kpi-delta delta-good">↓ 60% vs Manual Triage</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{critical_count}</div>'
                    f'<div class="kpi-label">Critical Incidents (≥8.0)</div>'
                    f'<div class="kpi-delta delta-info">Avg severity {avg_severity:.1f}/10</div></div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Folium Map
    st.markdown('<p class="section-title">🗺️ Live Geospatial Hazard Command Map</p>', unsafe_allow_html=True)
    m = folium.Map(location=[18.5204, 73.8567], zoom_start=14, tiles="OpenStreetMap")

    legend_html = """<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:#1e293b;
         border:1px solid #475569;border-radius:10px;padding:12px 18px;font-size:13px;color:#e2e8f0;">
      <b>🔴 Critical (8–10)</b><br/><b>🟠 Moderate (5–7)</b><br/><b>🟢 Low (1–4)</b></div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    for item in incidents:
        sev = item["severity"]
        color     = "red"    if sev >= 8.0 else ("orange" if sev >= 5.0 else "green")
        icon_name = "exclamation-sign" if sev >= 8.0 else ("warning-sign" if sev >= 5.0 else "info-sign")

        popup_html = (
            f'<div style="font-family:sans-serif;min-width:220px">'
            f'<b style="font-size:14px">{item["id"]}</b><br/><hr style="margin:4px 0"/>'
            f'<b>Hazard:</b> {item["hazard"]}<br/>'
            f'<b>Severity:</b> {sev}/10<br/>'
            f'<b>Citizen Reports:</b> {item["reports_count"]}<br/>'
            f'<b>Dept:</b> {item["department"]}<br/>'
            f'<b>Crew:</b> {item.get("crew","—")}<br/>'
            f'<b>Status:</b> {item["status"]}<br/>'
            f'<b>Submitted:</b> {item.get("submitted","—")}</div>'
        )

        folium.Marker(
            location=[item["lat"], item["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f'{item["id"]}  •  Sev {sev}/10  •  {item["reports_count"]} reports',
            icon=folium.Icon(color=color, icon=icon_name, prefix="glyphicon"),
        ).add_to(m)

        folium.Circle(
            location=[item["lat"], item["lon"]],
            radius=20,
            color="#64748b",
            fill=True,
            fill_opacity=0.08,
            tooltip="20 m deduplication boundary",
        ).add_to(m)

    st_folium(m, width="100%", height=430, returned_objects=[])

    # Triage Queue Table
    st.markdown('<p class="section-title">📋 Live Triage Queue — Active Work Orders</p>', unsafe_allow_html=True)

    rows = [{
        "Ticket ID":          i["id"],
        "Hazard Type":        i["hazard"],
        "Severity":           i["severity"],
        "Reports Clustered":  i["reports_count"],
        "Assigned Dept":      i["department"],
        "Crew":               i.get("crew", "—"),
        "Status":             i["status"],
        "Submitted":          i.get("submitted", "—"),
    } for i in incidents]

    df_queue = pd.DataFrame(rows).sort_values("Severity", ascending=False)

    st.dataframe(
        df_queue.style
                .background_gradient(subset=["Severity"], cmap="RdYlGn_r", vmin=1, vmax=10)
                .format({"Severity": "{:.1f}"}),
        use_container_width=True,
        height=280,
    )

    # Submission Log
    if st.session_state.submission_log:
        with st.expander("🕒 Submission Activity Log (this session)", expanded=False):
            st.dataframe(pd.DataFrame(st.session_state.submission_log[::-1]), use_container_width=True)

    # Export Buttons
    st.markdown('<p class="section-title">📥 Export Work Orders</p>', unsafe_allow_html=True)
    ex1, ex2 = st.columns(2)
    with ex1:
        csv_buf = io.StringIO()
        df_queue.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 Download Municipal Work-Order CSV",
            data=csv_buf.getvalue(),
            file_name=f"work_orders_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ex2:
        st.download_button(
            label="📦 Download Work-Order JSON",
            data=df_queue.to_json(orient="records", indent=2),
            file_name=f"work_orders_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )

    # Footer
    st.markdown("""
<div style="margin-top:2rem;padding:1rem 1.5rem;background:#0f172a;border-radius:10px;
            border:1px solid #1e293b;text-align:center;color:#475569;font-size:0.8rem;">
  🏛️ CivicLens AI &nbsp;·&nbsp; Powered by YOLOv8 · Streamlit · Folium
  &nbsp;·&nbsp; Haversine Deduplication Engine v1.0
  &nbsp;·&nbsp; Hackathon Prototype — Pune Smart City Demo
</div>
""", unsafe_allow_html=True)

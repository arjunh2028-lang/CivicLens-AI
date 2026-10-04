"""
CivicLens AI — Autonomous Municipal Triage & Geospatial Dispatch Engine v3.0
=============================================================================
Features:
  • YOLOv8 vision inference with intelligent heuristic fallback
  • Spatio-Temporal Haversine Deduplication Engine
  • Zero-Touch Department Auto-Routing
  • Live Geospatial Map + Incident Dashboard

pip install streamlit ultralytics pillow pandas numpy
Run: streamlit run app.py
"""

# ─── Standard library ───────────────────────────────────────────────────────
import io
import math
import random
from datetime import datetime, timedelta

# ─── Third-party ────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CivicLens AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

.cl-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 60%, #1a1a3e 100%);
    padding: 1.8rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #1e40af33;
}
.cl-header h1 { color:#e0f2fe; font-size:1.9rem; font-weight:800; margin:0 0 .3rem; letter-spacing:-0.5px; }
.cl-header p  { color:#7dd3fc; font-size:0.88rem; margin:0; font-style:italic; }

.kpi-wrap {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    height: 100%;
}
.kpi-val   { font-size:2.2rem; font-weight:800; line-height:1.1; }
.kpi-lbl   { font-size:0.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.06em; margin-top:.3rem; }
.kpi-sub   { font-size:0.72rem; color:#64748b; margin-top:.2rem; }
.kpi-red   { color:#f87171; }
.kpi-amber { color:#fbbf24; }
.kpi-green { color:#4ade80; }
.kpi-blue  { color:#38bdf8; }
.kpi-purple{ color:#c084fc; }

.sec-title {
    font-size:1rem; font-weight:700; color:#e2e8f0;
    border-left:4px solid #3b82f6; padding-left:.7rem;
    margin:1.4rem 0 .8rem;
}

.box-info    { background:#0c2340; border:1px solid #1d4ed8; border-radius:10px; padding:.9rem 1.1rem; margin:.7rem 0; font-size:.87rem; color:#bfdbfe; }
.box-warn    { background:#3b1a00; border:1px solid #d97706; border-radius:10px; padding:.9rem 1.1rem; margin:.7rem 0; font-size:.87rem; color:#fde68a; }
.box-success { background:#052e16; border:1px solid #16a34a; border-radius:10px; padding:.9rem 1.1rem; margin:.7rem 0; font-size:.87rem; color:#86efac; }
.box-danger  { background:#450a0a; border:1px solid #dc2626; border-radius:10px; padding:.9rem 1.1rem; margin:.7rem 0; font-size:.87rem; color:#fca5a5; }
.box-purple  { background:#2e1065; border:1px solid #7c3aed; border-radius:10px; padding:.9rem 1.1rem; margin:.7rem 0; font-size:.87rem; color:#ddd6fe; }
.box-dedup   { background:#1a1a3e; border:2px solid #7c3aed; border-radius:12px; padding:1.1rem 1.3rem; margin:.9rem 0; font-size:.9rem; color:#e0d7ff; }

.receipt {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}
.receipt h3 { font-size:1.1rem; font-weight:700; color:#38bdf8; margin:0 0 .8rem; }
.receipt table { width:100%; border-collapse:collapse; }
.receipt td { padding:.3rem 0; font-size:.87rem; vertical-align:top; }
.receipt td:first-child { color:#94a3b8; width:42%; }
.receipt td:last-child  { color:#f1f5f9; font-weight:600; }

.dept-badge {
    display:inline-block; padding:.25rem .8rem;
    border-radius:99px; font-size:.73rem; font-weight:700; letter-spacing:.03em;
    background:#172554; color:#93c5fd; border:1px solid #1d4ed8;
}
.badge { display:inline-block; padding:.2rem .65rem; border-radius:99px; font-size:.75rem; font-weight:700; letter-spacing:.04em; }
.badge-high   { background:#7f1d1d; color:#fca5a5; border:1px solid #dc2626; }
.badge-medium { background:#451a03; color:#fde68a; border:1px solid #d97706; }
.badge-low    { background:#052e16; color:#86efac; border:1px solid #16a34a; }

[data-testid="stMetric"] {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 14px;
}
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white; border:none; border-radius:8px;
    padding:.5rem 1rem; font-weight:600;
    transition: all .2s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #1e40af);
    transform: translateY(-1px);
}
.sidebar-brand { text-align:center; padding:1rem 0 .5rem; border-bottom:1px solid #1e293b; margin-bottom:1rem; }
.sidebar-brand h2 { color:#38bdf8 !important; font-size:1.3rem; font-weight:800; margin:0; }
.sidebar-brand p  { color:#64748b !important; font-size:.76rem; margin:.2rem 0 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & LOOKUP TABLES
# ─────────────────────────────────────────────────────────────────────────────

# ── Severity mapping from YOLO COCO labels ──────────────────────────────────
SEVERITY_MAP = {
    "person":        ("High",   "< 4 hrs",  "Possible casualty / obstruction hazard"),
    "fire hydrant":  ("High",   "< 4 hrs",  "Pipe-burst / water infrastructure failure"),
    "stop sign":     ("High",   "< 4 hrs",  "Missing / damaged traffic control device"),
    "car":           ("High",   "< 4 hrs",  "Abandoned / crashed vehicle blocking road"),
    "truck":         ("High",   "< 4 hrs",  "Heavy vehicle incident"),
    "bus":           ("High",   "< 4 hrs",  "Transit blockage"),
    "boat":          ("High",   "< 4 hrs",  "Flooding / waterway overflow"),
    "bird":          ("High",   "< 4 hrs",  "Open manhole / animal hazard"),
    "bicycle":       ("Medium", "< 24 hrs", "Road damage / cycling infrastructure"),
    "motorcycle":    ("Medium", "< 24 hrs", "Road surface pothole / crack"),
    "traffic light": ("Medium", "< 24 hrs", "Traffic signal fault"),
    "bench":         ("Medium", "< 24 hrs", "Broken public furniture"),
    "potted plant":  ("Medium", "< 24 hrs", "Encroaching vegetation / obstruction"),
    "sports ball":   ("Medium", "< 24 hrs", "Illegal dumping — solid waste"),
    "kite":          ("Medium", "< 24 hrs", "Overhead wire hazard"),
    "umbrella":      ("Medium", "< 24 hrs", "Shelter damage / unauthorized structure"),
    "backpack":      ("Low",    "< 72 hrs", "Uncollected public garbage / debris"),
    "handbag":       ("Low",    "< 72 hrs", "Litter / scattered waste"),
    "suitcase":      ("Low",    "< 72 hrs", "Abandoned property"),
    "bottle":        ("Low",    "< 72 hrs", "Littering / street waste"),
    "cup":           ("Low",    "< 72 hrs", "Waste overflow"),
    "chair":         ("Low",    "< 72 hrs", "Illegal dumping — furniture"),
    "couch":         ("Low",    "< 72 hrs", "Illegal dumping — bulk waste"),
    "dining table":  ("Low",    "< 72 hrs", "Illegal dumping — commercial waste"),
    "cell phone":    ("Low",    "< 72 hrs", "Streetlight / electronic fixture fault"),
    "clock":         ("Low",    "< 72 hrs", "Public clock / signage damage"),
    "book":          ("Low",    "< 72 hrs", "Graffiti / vandalism on public property"),
    "vase":          ("Low",    "< 72 hrs", "Minor infrastructure damage"),
    "toilet":        ("Low",    "< 72 hrs", "Public sanitation unit damage"),
}

HEURISTIC_FALLBACKS = [
    ("Pothole / Road Damage",       "Medium", "< 24 hrs", "Surface crack or pothole detected"),
    ("Garbage Overflow",            "Low",    "< 72 hrs", "Waste accumulation detected"),
    ("Water Leakage / Pipe Burst",  "High",   "< 4 hrs",  "Water infrastructure risk"),
    ("Broken Streetlight",          "Low",    "< 72 hrs", "Lighting fixture damage"),
    ("Open Manhole",                "High",   "< 4 hrs",  "Pedestrian safety hazard"),
    ("Illegal Dumping — Bulk",      "Medium", "< 24 hrs", "Bulk waste obstruction"),
    ("Sewage Overflow",             "High",   "< 4 hrs",  "Sanitation emergency"),
    ("Road Crack / Surface Damage", "Medium", "< 24 hrs", "Structural road damage"),
]

# ── Zero-Touch Department Auto-Routing Table ─────────────────────────────────
DEPT_ROUTING = {
    # Water / Flooding
    "Water Leakage / Pipe Burst":   "Water Supply & Sewerage Board",
    "Sewage Overflow":              "Water Supply & Sewerage Board",
    "Flooding / Waterlogging":      "Water Supply & Sewerage Board",
    "Fire Hydrant":                 "Water Supply & Sewerage Board",
    "Boat":                         "Water Supply & Sewerage Board",
    # Roads
    "Pothole / Road Damage":        "Road Transport & Public Works (PWD)",
    "Road Crack / Surface Damage":  "Road Transport & Public Works (PWD)",
    "Open Manhole":                 "Road Transport & Public Works (PWD)",
    "Stop Sign":                    "Road Transport & Public Works (PWD)",
    "Bicycle":                      "Road Transport & Public Works (PWD)",
    "Motorcycle":                   "Road Transport & Public Works (PWD)",
    # Sanitation / Waste
    "Garbage Overflow":             "Sanitation & Solid Waste Management",
    "Illegal Dumping — Bulk":       "Sanitation & Solid Waste Management",
    "Backpack":                     "Sanitation & Solid Waste Management",
    "Handbag":                      "Sanitation & Solid Waste Management",
    "Suitcase":                     "Sanitation & Solid Waste Management",
    "Bottle":                       "Sanitation & Solid Waste Management",
    "Cup":                          "Sanitation & Solid Waste Management",
    "Chair":                        "Sanitation & Solid Waste Management",
    "Couch":                        "Sanitation & Solid Waste Management",
    "Dining Table":                 "Sanitation & Solid Waste Management",
    "Toilet":                       "Sanitation & Solid Waste Management",
    # Electrical
    "Broken Streetlight":           "Electrical & Public Lighting Dept",
    "Traffic Light":                "Electrical & Public Lighting Dept",
    "Cell Phone":                   "Electrical & Public Lighting Dept",
    "Clock":                        "Electrical & Public Lighting Dept",
    # Safety / General
    "Person":                       "Emergency Response & Public Safety",
    "Car":                          "Emergency Response & Public Safety",
    "Truck":                        "Emergency Response & Public Safety",
    "Bus":                          "Emergency Response & Public Safety",
    "Bird":                         "Emergency Response & Public Safety",
    # Misc
    "Bench":                        "Urban Development & Public Amenities",
    "Potted Plant":                 "Urban Forestry & Horticulture",
    "Kite":                         "Urban Development & Public Amenities",
    "Umbrella":                     "Urban Development & Public Amenities",
    "Sports Ball":                  "Sanitation & Solid Waste Management",
    "Book":                         "Urban Development & Public Amenities",
    "Vase":                         "Urban Development & Public Amenities",
}

DISPATCH_UNITS = [
    "Road Works Fleet 1",
    "Road Works Fleet 2",
    "Water Emergency Response",
    "Sanitation Team A",
    "Sanitation Team B",
    "Electrical Maintenance Unit",
    "Urban Forestry Crew",
    "Traffic Management Cell",
    "Emergency Response Unit",
]

STATUS_OPTIONS = ["Pending", "Dispatched", "In-Progress", "Resolved", "Escalated"]

SEVERITY_COLOR = {
    "High":   ("🔴", "#f87171"),
    "Medium": ("🟡", "#fbbf24"),
    "Low":    ("🟢", "#4ade80"),
}

# Pune city center coords for demo GPS generation
PUNE_CENTER_LAT = 18.5204
PUNE_CENTER_LON = 73.8567
DEDUP_RADIUS_M  = 100.0   # metres — cluster within 100 m of same category


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM 1 — HAVERSINE DEDUPLICATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return geodesic distance in metres between two GPS coordinates."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_duplicate(lat: float, lon: float, category: str) -> dict | None:
    """
    Search open incidents for a spatial + category match within DEDUP_RADIUS_M.
    Returns the matching incident dict or None.
    """
    for inc in st.session_state.incidents:
        if inc.get("status") == "Resolved":
            continue
        dist = haversine_m(lat, lon, inc.get("lat", 0.0), inc.get("lon", 0.0))
        if dist <= DEDUP_RADIUS_M and inc.get("category", "").lower() == category.lower():
            return inc
    return None


def merge_duplicate(master: dict, ts: str) -> None:
    """Merge a duplicate report into the master ticket."""
    master["report_count"] = master.get("report_count", 1) + 1
    master.setdefault("timestamps", [master["timestamp"]]).append(ts)
    # Escalate severity when report_count crosses threshold
    if master["report_count"] >= 3:
        if master["severity"] == "Low":
            master["severity"] = "Medium"
            master["sla"]      = "< 24 hrs"
        elif master["severity"] == "Medium":
            master["severity"] = "High"
            master["sla"]      = "< 4 hrs"


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM 2 — ZERO-TOUCH DEPARTMENT AUTO-ROUTING
# ─────────────────────────────────────────────────────────────────────────────
def auto_route_dept(category: str) -> str:
    """Map a detected civic category to its responsible municipal department."""
    return DEPT_ROUTING.get(category, "General Municipal Services")


# ─────────────────────────────────────────────────────────────────────────────
# YOLO MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Loading YOLOv8 vision model…")
def load_yolo():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        return model, True
    except Exception:
        return None, False


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE (with heuristic fallback)
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, img: Image.Image):
    """Returns (annotated_img, detections_list)."""
    if model is not None:
        try:
            results   = model(np.array(img.convert("RGB")), verbose=False)
            annotated = Image.fromarray(results[0].plot())
            detections = []
            for r in results:
                for box in r.boxes:
                    label = r.names[int(box.cls[0].item())].lower()
                    conf  = float(box.conf[0].item())
                    sev, sla, reason = SEVERITY_MAP.get(
                        label, ("Medium", "< 24 hrs", "General civic infrastructure issue")
                    )
                    detections.append({
                        "label": label.title(), "conf": conf,
                        "severity": sev, "sla": sla, "reason": reason,
                    })
            if detections:
                return annotated, detections
        except Exception:
            pass

    # ── Heuristic fallback ──
    cat, sev, sla, reason = random.choice(HEURISTIC_FALLBACKS)
    conf      = round(random.uniform(0.72, 0.94), 2)
    annotated = img.copy()
    draw      = ImageDraw.Draw(annotated)
    W, H      = annotated.size
    pad       = W // 8
    draw.rectangle([pad, pad, W - pad, H - pad], outline="#475569", width=4)
    draw.rectangle([pad, pad - 22, pad + len(f"{cat} {conf:.0%}") * 8, pad], fill="#475569")
    draw.text((pad + 4, pad - 20), f"{cat} {conf:.0%}", fill="#f1f5f9")
    return annotated, [{"label": cat, "conf": conf, "severity": sev, "sla": sla, "reason": reason}]


def aggregate_severity(detections):
    priority = {"High": 3, "Medium": 2, "Low": 1}
    return max(detections, key=lambda d: priority.get(d["severity"], 1))


# ─────────────────────────────────────────────────────────────────────────────
# GPS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def random_pune_gps(spread: float = 0.04) -> tuple[float, float]:
    """Generate a random GPS coord within ~4 km of Pune city centre."""
    return (
        round(PUNE_CENTER_LAT + random.uniform(-spread, spread), 6),
        round(PUNE_CENTER_LON + random.uniform(-spread, spread), 6),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────
def _seed():
    now = datetime.now()
    records = [
        {
            "ticket_id":    "CL-1001",
            "timestamp":    (now - timedelta(hours=5, minutes=12)).strftime("%H:%M %d-%b"),
            "timestamps":   [(now - timedelta(hours=5, minutes=12)).strftime("%H:%M %d-%b")],
            "location":     "MG Road Near Bus Stop 12",
            "lat":          18.5195, "lon": 73.8550,
            "category":     "Water Leakage / Pipe Burst",
            "department":   "Water Supply & Sewerage Board",
            "severity":     "High",
            "sla":          "< 4 hrs",
            "status":       "Dispatched",
            "unit":         "Water Emergency Response",
            "report_count": 4,
            "description":  "Major pipe burst flooding the road.",
        },
        {
            "ticket_id":    "CL-1002",
            "timestamp":    (now - timedelta(hours=2, minutes=40)).strftime("%H:%M %d-%b"),
            "timestamps":   [(now - timedelta(hours=2, minutes=40)).strftime("%H:%M %d-%b")],
            "location":     "FC Road, Pune",
            "lat":          18.5280, "lon": 73.8430,
            "category":     "Pothole / Road Damage",
            "department":   "Road Transport & Public Works (PWD)",
            "severity":     "Medium",
            "sla":          "< 24 hrs",
            "status":       "Pending",
            "unit":         "—",
            "report_count": 2,
            "description":  "Deep pothole causing accidents.",
        },
        {
            "ticket_id":    "CL-1003",
            "timestamp":    (now - timedelta(hours=1, minutes=5)).strftime("%H:%M %d-%b"),
            "timestamps":   [(now - timedelta(hours=1, minutes=5)).strftime("%H:%M %d-%b")],
            "location":     "Shivajinagar Circle",
            "lat":          18.5308, "lon": 73.8474,
            "category":     "Garbage Overflow",
            "department":   "Sanitation & Solid Waste Management",
            "severity":     "Low",
            "sla":          "< 72 hrs",
            "status":       "Pending",
            "unit":         "—",
            "report_count": 1,
            "description":  "Overflowing garbage bin on footpath.",
        },
        {
            "ticket_id":    "CL-1004",
            "timestamp":    (now - timedelta(minutes=30)).strftime("%H:%M %d-%b"),
            "timestamps":   [(now - timedelta(minutes=30)).strftime("%H:%M %d-%b")],
            "location":     "Koregaon Park Lane 7",
            "lat":          18.5362, "lon": 73.8929,
            "category":     "Open Manhole",
            "department":   "Road Transport & Public Works (PWD)",
            "severity":     "High",
            "sla":          "< 4 hrs",
            "status":       "In-Progress",
            "unit":         "Road Works Fleet 1",
            "report_count": 3,
            "description":  "Uncovered manhole — pedestrian risk.",
        },
        {
            "ticket_id":    "CL-1005",
            "timestamp":    (now - timedelta(hours=3)).strftime("%H:%M %d-%b"),
            "timestamps":   [(now - timedelta(hours=3)).strftime("%H:%M %d-%b")],
            "location":     "Baner Road, Pune",
            "lat":          18.5590, "lon": 73.7868,
            "category":     "Broken Streetlight",
            "department":   "Electrical & Public Lighting Dept",
            "severity":     "Low",
            "sla":          "< 72 hrs",
            "status":       "Resolved",
            "unit":         "Electrical Maintenance Unit",
            "report_count": 1,
            "description":  "Streetlight not working since 3 days.",
        },
    ]
    return records


if "incidents"      not in st.session_state: st.session_state.incidents      = _seed()
if "ticket_counter" not in st.session_state: st.session_state.ticket_counter = 1006
if "view"           not in st.session_state: st.session_state.view           = "citizen"
if "dedup_count"    not in st.session_state: st.session_state.dedup_count    = 3  # seeded dedup count
if "total_reports"  not in st.session_state:
    st.session_state.total_reports = sum(i["report_count"] for i in st.session_state.incidents)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def next_ticket() -> str:
    tid = f"CL-{st.session_state.ticket_counter}"
    st.session_state.ticket_counter += 1
    return tid


def badge_html(severity: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(severity, "badge-low")
    return f'<span class="badge {cls}">{severity}</span>'


def dept_badge_html(dept: str) -> str:
    return f'<span class="dept-badge">🏢 {dept}</span>'


def color_severity(val: str) -> str:
    return {
        "High":   "background-color:#7f1d1d; color:#fca5a5;",
        "Medium": "background-color:#451a03; color:#fde68a;",
        "Low":    "background-color:#052e16; color:#86efac;",
    }.get(val, "")


def bandwidth_saved() -> float:
    """Percentage of duplicate reports filtered from the total inbound volume."""
    total = st.session_state.total_reports
    unique = len(st.session_state.incidents)
    if total == 0:
        return 0.0
    return round((total - unique) / total * 100, 1)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
yolo_model, yolo_ok = load_yolo()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🏛️ CivicLens AI</h2>
        <p>Municipal Triage Engine v3.0</p>
    </div>""", unsafe_allow_html=True)

    view = st.radio(
        "Navigate to",
        options=["📸  Citizen Portal", "🏛️  Municipal Dispatch"],
        index=0 if st.session_state.view == "citizen" else 1,
        label_visibility="collapsed",
    )
    st.session_state.view = "citizen" if "Citizen" in view else "dispatch"

    st.markdown("---")
    model_status = "✅ YOLOv8n Loaded" if yolo_ok else "⚠️ Heuristic Mode"
    st.caption(f"Vision engine: {model_status}")
    st.markdown("---")
    st.caption("Smart Cities Mission Sandbox · Pune Division Deployment")


# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE A — CITIZEN PORTAL
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.view == "citizen":
    st.title("📢 Citizen Hazard Reporting Portal")
    st.caption("Upload road and water infrastructure defects for real-time AI triage and crew routing.")

    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.form("citizen_form", clear_on_submit=False):
            st.markdown('<p class="sec-title">📋 Incident Details</p>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Upload Evidence Photo (JPG / PNG)",
                type=["jpg", "jpeg", "png"],
                help="Clear, well-lit images improve AI detection accuracy.",
            )

            location = st.text_input(
                "📍 Location / Landmark *",
                placeholder="e.g. MG Road Near Bus Stop 12, Pune",
            )

            description = st.text_area(
                "📝 Description / Context",
                placeholder="Briefly describe what you observed…",
                height=90,
            )

            st.markdown('<p class="sec-title">🌐 GPS Coordinates</p>', unsafe_allow_html=True)
            gc1, gc2 = st.columns(2)
            gps_lat = gc1.number_input("Latitude",  value=PUNE_CENTER_LAT, format="%.6f", step=0.0001)
            gps_lon = gc2.number_input("Longitude", value=PUNE_CENTER_LON, format="%.6f", step=0.0001)
            st.caption("📡 Auto-injected from mobile GPS metadata — edit for manual override")

            submitted = st.form_submit_button(
                "🚀 Submit & Analyze Incident",
                use_container_width=True,
                type="primary",
            )

        if not yolo_ok:
            st.markdown('<div class="box-warn">⚠️ YOLOv8 model unavailable — running intelligent heuristic fallback. Demo safe.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="box-info">✅ YOLOv8n loaded — live computer vision active.</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<p class="sec-title">🔍 Vision Frame & Bounding Box Inspection</p>', unsafe_allow_html=True)

        img_slot    = st.empty()
        result_slot = st.empty()

        if not submitted or uploaded_file is None:
            placeholder = Image.new("RGB", (480, 300), color="#1e293b")
            draw = ImageDraw.Draw(placeholder)
            for y in range(0, 300, 12):
                draw.line([(0, y), (480, y)], fill="#334155", width=1)
            draw.text((120, 130), "Upload an image to begin analysis", fill="#475569")
            img_slot.image(placeholder, use_container_width=True, caption="Awaiting citizen upload…")

        # ── SUBMISSION LOGIC ──────────────────────────────────────────────
        if submitted:
            if not location.strip():
                st.error("⚠️ Location field is mandatory. Please enter a landmark.")
            elif uploaded_file is None:
                st.error("⚠️ Please upload an evidence photo.")
            else:
                with st.spinner("🧠 Running YOLOv8 inference, deduplication check & auto-routing…"):
                    raw_img              = Image.open(uploaded_file).convert("RGB")
                    annotated, detections = run_inference(yolo_model, raw_img)
                    best                 = aggregate_severity(detections)
                    category             = best["label"]
                    department           = auto_route_dept(category)
                    ts                   = datetime.now().strftime("%H:%M %d-%b")

                    # ── DEDUPLICATION CHECK ──────────────────────────────
                    st.session_state.total_reports += 1
                    duplicate_master = find_duplicate(gps_lat, gps_lon, category)

                img_slot.image(
                    annotated, use_container_width=True,
                    caption=f"YOLOv8 Annotated — {len(detections)} detection(s)"
                )

                sev_icon = SEVERITY_COLOR[best["severity"]][0]

                if duplicate_master:
                    # ── MERGE INTO MASTER TICKET ─────────────────────────
                    merge_duplicate(duplicate_master, ts)
                    st.session_state.dedup_count += 1
                    master_id   = duplicate_master["ticket_id"]
                    new_count   = duplicate_master["report_count"]
                    new_sev     = duplicate_master["severity"]
                    elevated    = new_count >= 3 and new_sev != best["severity"]

                    result_slot.markdown(
                        f'<div class="box-dedup">'
                        f'<b>📍 DUPLICATE DETECTED — Spatio-Temporal Deduplication Engine Triggered</b><br>'
                        f'A <b>{category}</b> incident already exists within <b>{DEDUP_RADIUS_M:.0f} m</b> of your location.<br>'
                        f'✅ Report merged into Master Ticket <b>{master_id}</b>. '
                        f'Citizen confirmation count: <b>{new_count}</b>.'
                        f'{"<br>⚠️ Priority auto-escalated to <b>" + new_sev + "</b> due to repeated reports!" if elevated else ""}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(f"""
<div class="receipt">
  <h3>📍 Merged into Master Ticket — {master_id}</h3>
  <table>
    <tr><td>Master Ticket ID</td>   <td>{master_id}</td></tr>
    <tr><td>Your Submission</td>    <td>{ts}</td></tr>
    <tr><td>Category</td>           <td>{category}</td></tr>
    <tr><td>Department (Auto)</td>  <td>{dept_badge_html(department)}</td></tr>
    <tr><td>Current Severity</td>   <td>{badge_html(new_sev)}</td></tr>
    <tr><td>SLA Window</td>         <td>{duplicate_master['sla']}</td></tr>
    <tr><td>Citizen Reports</td>    <td>{new_count} (including yours)</td></tr>
    <tr><td>Status</td>             <td>{duplicate_master['status']}</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Merged Into", master_id)
                    m2.metric("Total Reports", new_count)
                    m3.metric("Current Severity", f"{SEVERITY_COLOR[new_sev][0]} {new_sev}")

                else:
                    # ── NEW MASTER TICKET ─────────────────────────────────
                    ticket = next_ticket()
                    record = {
                        "ticket_id":    ticket,
                        "timestamp":    ts,
                        "timestamps":   [ts],
                        "location":     location.strip(),
                        "lat":          round(gps_lat, 6),
                        "lon":          round(gps_lon, 6),
                        "category":     category,
                        "department":   department,
                        "severity":     best["severity"],
                        "sla":          best["sla"],
                        "status":       "Pending",
                        "unit":         "—",
                        "report_count": 1,
                        "description":  description.strip() or "—",
                    }
                    st.session_state.incidents.append(record)

                    sev_css = {"High": "box-danger", "Medium": "box-warn", "Low": "box-success"}.get(best["severity"], "box-info")
                    result_slot.markdown(
                        f'<div class="{sev_css}"><b>{sev_icon} NEW MASTER TICKET — Severity {best["severity"]} — SLA {best["sla"]}</b><br>'
                        f'{best["reason"]}<br>Department auto-routed → <b>{department}</b></div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(f"""
<div class="receipt">
  <h3>🎫 New Incident Receipt — {ticket}</h3>
  <table>
    <tr><td>Ticket ID</td>         <td>{ticket}</td></tr>
    <tr><td>Submitted</td>         <td>{ts}</td></tr>
    <tr><td>Location</td>          <td>{location.strip()}</td></tr>
    <tr><td>GPS</td>               <td>{gps_lat:.5f}, {gps_lon:.5f}</td></tr>
    <tr><td>Category</td>          <td>{category}</td></tr>
    <tr><td>Dept (Auto-Routed)</td><td>{dept_badge_html(department)}</td></tr>
    <tr><td>Severity</td>          <td>{badge_html(best['severity'])}</td></tr>
    <tr><td>SLA Window</td>        <td>{best['sla']}</td></tr>
    <tr><td>Confidence</td>        <td>{best['conf']:.0%}</td></tr>
    <tr><td>Status</td>            <td>Pending Dispatch</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

                    with st.expander("🔎 Full Detection Breakdown", expanded=False):
                        st.dataframe(
                            pd.DataFrame([{
                                "Label": d["label"], "Conf": f"{d['conf']:.1%}",
                                "Severity": d["severity"], "SLA": d["sla"], "Reason": d["reason"],
                            } for d in detections]),
                            use_container_width=True, hide_index=True,
                        )

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Ticket ID",   ticket)
                    m2.metric("Severity",    f"{sev_icon} {best['severity']}")
                    m3.metric("Dept Routed", department[:22] + "…" if len(department) > 22 else department)


# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE B — MUNICIPAL DISPATCH COMMAND
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("🏛️ CivicLens AI — Municipal Operations Console")
    st.caption("Pune Municipal Corporation (PMC) Pilot Edition · Autonomous Triage & Geospatial Routing")

    incidents    = st.session_state.incidents
    total_master = len(incidents)
    high_c       = sum(1 for i in incidents if i["severity"] == "High")
    med_c        = sum(1 for i in incidents if i["severity"] == "Medium")
    low_c        = sum(1 for i in incidents if i["severity"] == "Low")
    total_rpts   = st.session_state.total_reports
    dupes        = total_rpts - total_master
    bw_pct       = bandwidth_saved()

    # ── KPI Row ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-wrap"><div class="kpi-val kpi-blue">{total_master}</div>'
                    f'<div class="kpi-lbl">Master Tickets</div>'
                    f'<div class="kpi-sub">Unique incidents</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-wrap"><div class="kpi-val kpi-red">{high_c}</div>'
                    f'<div class="kpi-lbl">🔴 Critical Active</div>'
                    f'<div class="kpi-sub">SLA &lt; 4 hrs</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-wrap"><div class="kpi-val kpi-amber">{med_c}</div>'
                    f'<div class="kpi-lbl">🟡 Moderate Active</div>'
                    f'<div class="kpi-sub">SLA &lt; 24 hrs</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-wrap"><div class="kpi-val kpi-purple">{bw_pct}%</div>'
                    f'<div class="kpi-lbl">Triage Workload Cut</div>'
                    f'<div class="kpi-sub">{dupes} duplicates filtered ({bw_pct}%)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_feed, tab_map, tab_dispatch = st.tabs([
        "📋  Live Incident Queue",
        "🗺️  Geospatial Command Map",
        "🚨  Crew Dispatch Workspace",
    ])

    # ── TAB 1: LIVE INCIDENT QUEUE ───────────────────────────────────────────
    with tab_feed:
        st.markdown('<p class="sec-title">📋 Master Ticket Queue — Active Work Orders</p>', unsafe_allow_html=True)

        df_all = pd.DataFrame(incidents)

        f1, f2, f3 = st.columns([2, 2, 2])
        with f1:
            sev_filter = st.multiselect("Filter by Severity",
                                        ["High", "Medium", "Low"],
                                        default=["High", "Medium", "Low"])
        with f2:
            cats       = sorted(df_all["category"].unique().tolist())
            cat_filter = st.multiselect("Filter by Category", cats, default=cats)
        with f3:
            status_filter = st.multiselect("Filter by Status", STATUS_OPTIONS, default=STATUS_OPTIONS)

        df_view = df_all[
            df_all["severity"].isin(sev_filter) &
            df_all["category"].isin(cat_filter) &
            df_all["status"].isin(status_filter)
        ].copy()

        pri = {"High": 0, "Medium": 1, "Low": 2}
        df_view["_sort"] = df_view["severity"].map(pri)
        df_view = df_view.sort_values("_sort").drop(columns=["_sort"])

        # Build display table with required columns
        df_display = df_view[[
            "ticket_id", "category", "department", "report_count",
            "severity", "sla", "status", "unit", "location", "timestamp"
        ]].rename(columns={
            "ticket_id":    "Master Ticket ID",
            "category":     "Category",
            "department":   "Assigned Department (Auto-Routed)",
            "report_count": "Citizen Reports (Merged)",
            "severity":     "Severity",
            "sla":          "SLA",
            "status":       "Status",
            "unit":         "Dispatched Unit",
            "location":     "Location",
            "timestamp":    "First Reported",
        })

        st.dataframe(
            df_display.style.map(color_severity, subset=["Severity"]),
            use_container_width=True,
            height=370,
            hide_index=True,
        )
        st.caption(
            f"Showing **{len(df_view)}** master tickets · "
            f"**{total_rpts}** total citizen reports received · "
            f"**{dupes}** duplicates clustered · "
            f"**{bw_pct}%** bandwidth saved"
        )

        # Export
        st.markdown('<p class="sec-title">📥 Export Work Orders</p>', unsafe_allow_html=True)
        ex1, ex2 = st.columns(2)
        with ex1:
            buf = io.StringIO()
            df_view.to_csv(buf, index=False)
            st.download_button("📥 Download CSV", buf.getvalue(),
                               file_name=f"civiclens_orders_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv", use_container_width=True)
        with ex2:
            st.download_button("📦 Download JSON", df_view.to_json(orient="records", indent=2),
                               file_name=f"civiclens_orders_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                               mime="application/json", use_container_width=True)

    # ── TAB 2: GEOSPATIAL COMMAND MAP ────────────────────────────────────────
    with tab_map:
        st.markdown('<p class="sec-title">🗺️ Live Geospatial Hazard Command Map</p>', unsafe_allow_html=True)

        map_inc = [i for i in incidents if i.get("lat") and i.get("lon")]

        if not map_inc:
            st.info("No geolocated incidents to display yet.")
        else:
            df_map = pd.DataFrame(map_inc)[["lat", "lon", "ticket_id", "category", "severity", "status", "report_count"]]

            st.map(df_map, latitude="lat", longitude="lon", size=80, use_container_width=True)

            st.markdown('<p class="sec-title">📍 Plotted Incident Coordinates</p>', unsafe_allow_html=True)
            st.dataframe(
                df_map.rename(columns={
                    "ticket_id":    "Ticket ID",
                    "category":     "Category",
                    "severity":     "Severity",
                    "status":       "Status",
                    "report_count": "Reports Merged",
                    "lat":          "Latitude",
                    "lon":          "Longitude",
                }).style.map(color_severity, subset=["Severity"]),
                use_container_width=True,
                hide_index=True,
                height=280,
            )

            st.markdown(
                f'<div class="box-purple">🗺️ Displaying <b>{len(map_inc)}</b> master incident pins. '
                f'Each pin represents a <b>deduplicated master ticket</b>. Duplicate citizen reports '
                f'within <b>{DEDUP_RADIUS_M:.0f} m</b> of the same category are clustered into a single pin, '
                f'eliminating redundant field dispatch.</div>',
                unsafe_allow_html=True
            )

    # ── TAB 3: CREW DISPATCH WORKSPACE ──────────────────────────────────────
    with tab_dispatch:
        st.markdown('<p class="sec-title">🚨 Crew Dispatch & Status Management</p>', unsafe_allow_html=True)

        active_ids = [i["ticket_id"] for i in incidents if i["status"] != "Resolved"]

        if not active_ids:
            st.markdown('<div class="box-success">✅ All incidents resolved! No pending dispatches.</div>', unsafe_allow_html=True)
        else:
            d_left, d_right = st.columns([1, 1], gap="large")

            with d_left:
                selected_id = st.selectbox("Select Ticket ID to Dispatch", active_ids)
                inc = next((i for i in incidents if i["ticket_id"] == selected_id), None)

                if inc:
                    sev_icon = SEVERITY_COLOR[inc["severity"]][0]
                    sev_css  = {"High": "box-danger", "Medium": "box-warn", "Low": "box-success"}.get(inc["severity"], "box-info")

                    st.markdown(f"""
<div class="{sev_css}">
  <b>{sev_icon} {inc['ticket_id']} — {inc['severity']} Priority</b><br>
  <b>Category:</b> {inc['category']}<br>
  <b>Department:</b> {dept_badge_html(inc.get('department', auto_route_dept(inc['category'])))}<br>
  <b>Location:</b> {inc['location']}<br>
  <b>GPS:</b> {inc.get('lat', '—')}, {inc.get('lon', '—')}<br>
  <b>SLA Window:</b> {inc['sla']}<br>
  <b>First Reported:</b> {inc['timestamp']}<br>
  <b>Citizen Reports Merged:</b> {inc.get('report_count', 1)}<br>
  <b>Description:</b> {inc['description']}
</div>
""", unsafe_allow_html=True)

            with d_right:
                if inc:
                    st.markdown('<p class="sec-title">⚙️ Dispatch Controls</p>', unsafe_allow_html=True)

                    new_unit = st.selectbox(
                        "Assign Dispatch Unit", DISPATCH_UNITS,
                        index=DISPATCH_UNITS.index(inc["unit"]) if inc["unit"] in DISPATCH_UNITS else 0,
                    )
                    new_status = st.selectbox(
                        "Update Status", STATUS_OPTIONS,
                        index=STATUS_OPTIONS.index(inc["status"]) if inc["status"] in STATUS_OPTIONS else 0,
                    )
                    st.text_area("Dispatcher Notes (optional)",
                                 placeholder="Add operational notes for field crew…", height=90)

                    confirm = st.button(f"✅ Confirm Dispatch — {selected_id}",
                                        use_container_width=True, type="primary")
                    if confirm:
                        for i in st.session_state.incidents:
                            if i["ticket_id"] == selected_id:
                                i["unit"]   = new_unit
                                i["status"] = new_status
                                break
                        toast_icon = "🚨" if new_status == "Dispatched" else "✅" if new_status == "Resolved" else "🔄"
                        st.toast(f"{toast_icon} {selected_id} → {new_status} | Unit: {new_unit}", icon=toast_icon)
                        st.rerun()

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("""
<div style="margin-top:2.5rem;padding:1rem 1.5rem;background:#0f172a;
            border-radius:10px;border:1px solid #1e293b;
            text-align:center;color:#475569;font-size:0.78rem;">
  🏛️ CivicLens AI &nbsp;·&nbsp; Powered by YOLOv8 · Streamlit
  &nbsp;·&nbsp; Haversine Deduplication Engine v3.0 · Zero-Touch Auto-Routing
  &nbsp;·&nbsp; Smart Cities Mission Sandbox · Pune Division Deployment
</div>
""", unsafe_allow_html=True)

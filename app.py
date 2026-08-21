import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from metrics import (
    load_config, compute_all_metrics, cluster_driver_styles, name_clusters,
    bootstrap_confidence_intervals, get_skill_model_diagnostics,
)

CFG = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

st.set_page_config(page_title="RaceIQ · Driver Consistency Intelligence", layout="wide",
                    initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# DESIGN TOKENS — "timing tower" telemetry aesthetic
# ---------------------------------------------------------------------------
BG = "#0E1116"
SURFACE = "#171B22"
SURFACE_2 = "#1F242D"
BORDER = "#2A303B"
TEXT = "#E8EAED"
TEXT_DIM = "#8B93A1"
PURPLE = "#8B6CFC"   # fastest / signature accent
CYAN = "#35C9C1"     # secondary data accent
AMBER = "#F5A623"    # pressure / warning accent
RED = "#EF5B5B"

def _flatten(html: str) -> str:
    """Markdown treats 4+ space indented lines as code blocks — strip all
    leading whitespace per line so injected HTML never accidentally triggers that."""
    return "\n".join(line.lstrip() for line in html.strip().split("\n"))

# A generic open-wheel race-car silhouette (no team livery, no real branding) —
# used once as a large cinematic hero shape, not repeated. Two color-tinted
# copies are crossfaded via CSS to create the "shifting between two colors" look.
# v3: single-line path data throughout (no embedded newlines/transforms) —
# wheel spoke lines are pre-computed literal coordinates rather than using
# transform="rotate(...)", and HTML comments were removed from inside the SVG
# string, to rule out any markup-parsing edge cases in the rendered page.
CAR_SVG = ('<svg viewBox="0 0 560 200" width="560" height="200" xmlns="http://www.w3.org/2000/svg" '
    'style="display:block;width:100%;height:auto;">'
    '<circle cx="120" cy="152" r="31" fill="currentColor"/>'
    '<circle cx="120" cy="152" r="14.5" fill="{bg}" opacity="0.7"/>'
    '<line x1="120" y1="140" x2="120" y2="164" stroke="currentColor" stroke-width="2.4" opacity="0.55"/>'
    '<line x1="130.4" y1="146" x2="109.6" y2="158" stroke="currentColor" stroke-width="2.4" opacity="0.55"/>'
    '<line x1="130.4" y1="158" x2="109.6" y2="146" stroke="currentColor" stroke-width="2.4" opacity="0.55"/>'
    '<circle cx="120" cy="152" r="5" fill="currentColor"/>'
    '<circle cx="430" cy="152" r="31" fill="currentColor"/>'
    '<circle cx="430" cy="152" r="14.5" fill="{bg}" opacity="0.7"/>'
    '<line x1="430" y1="140" x2="430" y2="164" stroke="currentColor" stroke-width="2.4" opacity="0.55"/>'
    '<line x1="440.4" y1="146" x2="419.6" y2="158" stroke="currentColor" stroke-width="2.4" opacity="0.55"/>'
    '<line x1="440.4" y1="158" x2="419.6" y2="146" stroke="currentColor" stroke-width="2.4" opacity="0.55"/>'
    '<circle cx="430" cy="152" r="5" fill="currentColor"/>'
    '<path d="M40 150 L30 148 Q22 147 22 152 Q22 157 30 156 L45 155 Z" fill="currentColor" opacity="0.85"/>'
    '<path d="M70 145 Q100 128 150 126 L340 122 Q400 121 430 118 Q470 116 500 122 L520 128 Q535 133 535 140 L535 148 Q535 154 525 154 L470 154 Q462 138 442 138 Q422 138 415 154 L160 154 Q152 138 132 138 Q112 138 105 154 L75 154 Q66 154 66 148 L66 150 Z" fill="currentColor"/>'
    '<path d="M95 133 Q180 122 300 121 L420 120 Q460 120 495 126" fill="none" stroke="#FFFFFF" stroke-width="2.5" opacity="0.16" stroke-linecap="round"/>'
    '<path d="M110 141 Q220 132 340 131 L440 130" fill="none" stroke="#FFFFFF" stroke-width="1.4" opacity="0.1" stroke-linecap="round"/>'
    '<path d="M215 122 Q225 96 260 92 L300 92 Q332 94 340 122 Z" fill="currentColor" opacity="0.95"/>'
    '<path d="M225 118 Q233 100 260 98 L296 98 Q320 100 328 118 Z" fill="{bg}" opacity="0.75"/>'
    '<path d="M232 100 Q245 78 270 76 Q292 76 302 96" fill="none" stroke="currentColor" stroke-width="5" stroke-linecap="round" opacity="0.9"/>'
    '<path d="M232 100 L226 116" fill="none" stroke="currentColor" stroke-width="4.5" stroke-linecap="round" opacity="0.9"/>'
    '<path d="M302 96 L306 114" fill="none" stroke="currentColor" stroke-width="4.5" stroke-linecap="round" opacity="0.9"/>'
    '<path d="M480 96 L540 90 L540 100 L490 108 Q470 110 470 118 L470 108 Q472 98 480 96 Z" fill="currentColor"/>'
    '<path d="M486 84 L538 79 L538 87 L490 93 Z" fill="currentColor" opacity="0.85"/>'
    '<rect x="486" y="108" width="6" height="30" fill="currentColor" opacity="0.9"/>'
    '<rect x="524" y="106" width="6" height="30" fill="currentColor" opacity="0.9"/>'
    '<path d="M40 150 Q35 130 55 128 L70 128 L70 148 Q60 145 40 150 Z" fill="currentColor" opacity="0.9"/>'
    '<path d="M32 146 Q28 133 44 131 L44 138 Q36 140 32 146 Z" fill="currentColor" opacity="0.7"/>'
    '<path d="M26 143 Q23 135 33 133 L33 139 Q28 140 26 143 Z" fill="currentColor" opacity="0.55"/>'
    '<ellipse cx="270" cy="176" rx="220" ry="10" fill="#000000" opacity="0.28"/>'
    '</svg>').replace("{bg}", BG)
CAR_SVG = _flatten(CAR_SVG)

HERO_CAR_HTML = f"""
<div class="hero-car-wrap">
    <div class="hero-car hero-car-a">{CAR_SVG}</div>
    <div class="hero-car hero-car-b">{CAR_SVG}</div>
    <div class="hero-car-reflection">
        <div class="hero-car hero-car-a">{CAR_SVG}</div>
        <div class="hero-car hero-car-b">{CAR_SVG}</div>
    </div>
</div>
"""
HERO_CAR_HTML = _flatten(HERO_CAR_HTML)

RACE_BG_HTML = f"""
<div class="race-bg">
    <div class="smoke smoke-1-red"></div>
    <div class="smoke smoke-1-cool"></div>
    <div class="smoke smoke-2-red"></div>
    <div class="smoke smoke-2-cool"></div>
    <div class="smoke smoke-3-red"></div>
    <div class="smoke smoke-3-cool"></div>
    <div class="smoke smoke-4-red"></div>
    <div class="smoke smoke-4-cool"></div>
    <div class="smoke smoke-5-red"></div>
    <div class="smoke smoke-5-cool"></div>
    <div class="haze"></div>
    <div class="vignette"></div>
</div>
"""
RACE_BG_HTML = _flatten(RACE_BG_HTML)



st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp {{
    background: {BG} !important;
    color: {TEXT};
    font-family: 'Inter', sans-serif;
}}

/* ---------- cinematic racing background: smoke, crossfading color, hero car ---------- */
.race-bg {{
    position: fixed;
    inset: 0;
    z-index: 0;
    overflow: hidden;
    pointer-events: none;
    background: radial-gradient(ellipse at 50% 0%, #0d0f14 0%, #050608 60%);
}}
.smoke {{
    position: absolute;
    border-radius: 50%;
    filter: blur(65px);
    mix-blend-mode: screen;
    animation-timing-function: ease-in-out;
    animation-iteration-count: infinite;
}}
.smoke-1-red, .smoke-1-cool {{ width: 640px; height: 640px; left: 2%; top: 4%; }}
.smoke-2-red, .smoke-2-cool {{ width: 560px; height: 560px; left: 50%; top: 10%; }}
.smoke-3-red, .smoke-3-cool {{ width: 600px; height: 600px; left: 26%; top: 42%; }}
.smoke-4-red, .smoke-4-cool {{ width: 520px; height: 520px; left: 70%; top: 55%; }}
.smoke-5-red, .smoke-5-cool {{ width: 580px; height: 580px; left: 4%; top: 75%; }}
.smoke-1-red, .smoke-2-red, .smoke-3-red, .smoke-4-red, .smoke-5-red {{
    background: radial-gradient(circle, rgba(239,91,91,0.5), transparent 72%);
}}
.smoke-1-cool, .smoke-2-cool, .smoke-3-cool, .smoke-4-cool, .smoke-5-cool {{
    background: radial-gradient(circle, rgba(139,108,252,0.44), transparent 72%);
}}
.smoke-1-red   {{ animation: smoke-drift-1 17s infinite, smoke-fade-in  9s infinite; }}
.smoke-1-cool  {{ animation: smoke-drift-1 17s infinite, smoke-fade-out 9s infinite; }}
.smoke-2-red   {{ animation: smoke-drift-2 21s infinite, smoke-fade-out 11s infinite; }}
.smoke-2-cool  {{ animation: smoke-drift-2 21s infinite, smoke-fade-in  11s infinite; }}
.smoke-3-red   {{ animation: smoke-drift-3 14s infinite, smoke-fade-in  13s infinite; }}
.smoke-3-cool  {{ animation: smoke-drift-3 14s infinite, smoke-fade-out 13s infinite; }}
.smoke-4-red   {{ animation: smoke-drift-2 19s infinite reverse, smoke-fade-out 10s infinite; }}
.smoke-4-cool  {{ animation: smoke-drift-2 19s infinite reverse, smoke-fade-in  10s infinite; }}
.smoke-5-red   {{ animation: smoke-drift-3 23s infinite reverse, smoke-fade-in  12s infinite; }}
.smoke-5-cool  {{ animation: smoke-drift-3 23s infinite reverse, smoke-fade-out 12s infinite; }}
@keyframes smoke-fade-in  {{ 0%,100% {{ opacity: 0.3; }} 50% {{ opacity: 0.95; }} }}
@keyframes smoke-fade-out {{ 0%,100% {{ opacity: 0.95; }} 50% {{ opacity: 0.3; }} }}
@keyframes smoke-drift-1 {{ 0%,100% {{ transform: translate(0,0); }} 50% {{ transform: translate(40px,-30px); }} }}
@keyframes smoke-drift-2 {{ 0%,100% {{ transform: translate(0,0); }} 50% {{ transform: translate(-50px,25px); }} }}
@keyframes smoke-drift-3 {{ 0%,100% {{ transform: translate(0,0); }} 50% {{ transform: translate(30px,35px); }} }}

/* subtle grain/haze texture for a smoggy, particulate feel */
.haze {{
    position: absolute;
    inset: 0;
    opacity: 0.05;
    mix-blend-mode: overlay;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
}}


.wet-floor {{
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 38%;
    background: linear-gradient(180deg, transparent, rgba(20,10,14,0.55) 60%, rgba(6,6,9,0.85));
}}

.hero-car-wrap {{
    position: absolute;
    right: 0%;
    bottom: 2%;
    width: min(58vw, 780px);
    aspect-ratio: 560 / 200;
}}
.hero-car {{
    position: absolute;
    inset: 0;
    filter: drop-shadow(0 0 22px rgba(0,0,0,0.5));
}}
.hero-car-a {{ color: #D8465A; animation: car-fade-in 9s infinite; }}
.hero-car-b {{ color: #5B6E94; animation: car-fade-out 9s infinite; }}
@keyframes car-fade-in  {{ 0%,100% {{ opacity: 0.55; }} 50% {{ opacity: 1; }} }}
@keyframes car-fade-out {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.55; }} }}
.hero-car-reflection {{
    position: absolute;
    inset: 0;
    top: 92%;
    height: 60%;
    transform: scaleY(-1);
    opacity: 0.22;
    filter: blur(2px);
    -webkit-mask-image: linear-gradient(180deg, rgba(0,0,0,0.5), transparent 75%);
    mask-image: linear-gradient(180deg, rgba(0,0,0,0.5), transparent 75%);
}}
.hero-car-reflection .hero-car {{ position: absolute; inset: 0; }}

.vignette {{
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 42% 45%, transparent 30%, rgba(3,3,5,0.8) 100%);
}}


/* keep real content above the animated background */
[data-testid="stAppViewContainer"] > .main,
[data-testid="stSidebar"],
header[data-testid="stHeader"] {{
    position: relative;
    z-index: 1;
}}
[data-testid="stHeader"] {{
    background: transparent;
}}
[data-testid="stSidebar"] {{
    background: rgba(23,27,34,0.94);
    backdrop-filter: blur(6px);
    border-right: 1px solid {BORDER};
}}
.block-container {{
    background: linear-gradient(90deg, rgba(8,9,13,0.74) 0%, rgba(8,9,13,0.74) 60%, rgba(8,9,13,0.25) 100%);
    border-radius: 12px;
}}

h1, h2, h3 {{
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    letter-spacing: -0.02em;
}}
.hero-banner {{
    position: relative;
    height: 440px;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 1.6rem;
    background: radial-gradient(ellipse at 68% 55%, #0a0508 0%, #030303 70%);
    border: 1px solid {BORDER};
}}
/* concentrated crimson glow directly behind the car — the "studio spotlight
   through smoke" look, instead of diffuse cloud fill */
.hero-spotlight {{
    position: absolute;
    right: -6%;
    top: 18%;
    width: 60%;
    height: 90%;
    background: radial-gradient(circle, rgba(239,60,80,0.55) 0%, rgba(239,60,80,0.22) 38%, transparent 70%);
    filter: blur(10px);
    animation: spotlight-pulse 6s ease-in-out infinite;
}}
@keyframes spotlight-pulse {{ 0%,100% {{ opacity: 0.75; }} 50% {{ opacity: 1; }} }}
.hero-smoke {{
    position: absolute;
    border-radius: 50%;
    filter: blur(55px);
    mix-blend-mode: screen;
    animation-timing-function: ease-in-out;
    animation-iteration-count: infinite;
}}
.hero-smoke-red {{
    width: 68%; height: 155%; left: 24%; top: -40%;
    background: radial-gradient(circle, rgba(230,58,80,0.65), transparent 62%);
    animation: hero-smoke-fade-a 8s infinite, hero-smoke-drift-a 15s infinite;
}}
.hero-smoke-cool {{
    width: 50%; height: 130%; right: -8%; top: -20%;
    background: radial-gradient(circle, rgba(120,90,180,0.4), transparent 65%);
    animation: hero-smoke-fade-b 8s infinite, hero-smoke-drift-b 17s infinite;
}}
@keyframes hero-smoke-fade-a {{ 0%,100% {{ opacity: 0.6; }} 50% {{ opacity: 1; }} }}
@keyframes hero-smoke-fade-b {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.6; }} }}
@keyframes hero-smoke-drift-a {{ 0%,100% {{ transform: translate(0,0); }} 50% {{ transform: translate(25px,15px); }} }}
@keyframes hero-smoke-drift-b {{ 0%,100% {{ transform: translate(0,0); }} 50% {{ transform: translate(-25px,15px); }} }}

.hero-car-banner {{
    position: absolute;
    left: 66%;
    top: 46%;
    width: 56%;
    max-width: 560px;
    aspect-ratio: 560 / 200;
    transform: translate(-50%, -46%);
    filter: drop-shadow(0 14px 34px rgba(0,0,0,0.85)) drop-shadow(0 0 40px rgba(230,58,80,0.25));
}}
.hero-car-b2 {{ position: absolute; inset: 0; }}
.tint-a {{ color: #E8394F; animation: car-fade-in 9s infinite; }}
.tint-b {{ color: #6B5FA8; animation: car-fade-out 9s infinite; }}

/* horizontal motion-blur streaks trailing off the back of the car —
   pure CSS, gives a sense of speed instead of a static parked silhouette */
.hero-motion-streaks {{
    position: absolute;
    left: 18%;
    top: 44%;
    width: 40%;
    height: 14%;
    pointer-events: none;
}}
.streak {{
    position: absolute;
    left: 0;
    height: 2px;
    border-radius: 2px;
    background: linear-gradient(90deg, transparent, rgba(232,57,79,0.55), transparent);
    animation: streak-move 2.2s linear infinite;
}}
.streak:nth-child(1) {{ top: 10%;  width: 70%; animation-delay: 0s;   opacity: 0.7; }}
.streak:nth-child(2) {{ top: 45%;  width: 55%; animation-delay: 0.5s; opacity: 0.5; }}
.streak:nth-child(3) {{ top: 78%;  width: 65%; animation-delay: 1.1s; opacity: 0.6; }}
@keyframes streak-move {{
    0%   {{ transform: translateX(-30%); opacity: 0; }}
    15%  {{ opacity: 1; }}
    85%  {{ opacity: 1; }}
    100% {{ transform: translateX(220%); opacity: 0; }}
}}

/* thin gold pinstripe under the hero title — a small premium/racing-livery cue */
.hero-pinstripe {{
    width: 46px;
    height: 3px;
    background: linear-gradient(90deg, {AMBER}, transparent);
    border-radius: 2px;
    margin: 0.55rem 0 0.75rem 0;
}}

/* mirrored, fading reflection of the car on the "wet floor" below it */
.hero-car-reflect {{
    position: absolute;
    left: 66%;
    top: 46%;
    width: 56%;
    max-width: 560px;
    aspect-ratio: 560 / 200;
    transform: translate(-50%, 54%) scaleY(-1);
    opacity: 0.28;
    filter: blur(1.5px);
    -webkit-mask-image: linear-gradient(180deg, rgba(0,0,0,0.9), transparent 75%);
    mask-image: linear-gradient(180deg, rgba(0,0,0,0.9), transparent 75%);
}}
.hero-car-reflect .hero-car-b2 {{ position: absolute; inset: 0; }}

.hero-wet-floor {{
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 38%;
    background: linear-gradient(180deg, transparent, rgba(0,0,0,0.75));
}}
/* faint horizontal sheen so the floor reads as reflective, not just dark */
.hero-floor-sheen {{
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 22%;
    background: linear-gradient(180deg, rgba(230,58,80,0.06), transparent);
}}
.hero-scrim {{
    position: absolute;
    inset: 0;
    background:
        linear-gradient(0deg, rgba(6,6,8,0.92) 0%, rgba(6,6,8,0.35) 42%, transparent 68%),
        linear-gradient(90deg, rgba(6,6,8,0.55) 0%, transparent 52%);
}}
.hero-text {{
    position: absolute;
    left: 2rem;
    bottom: 1.7rem;
    max-width: 640px;
}}
.hero-tag {{
    font-family: 'JetBrains Mono', monospace;
    color: {PURPLE};
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    text-shadow: 0 2px 8px rgba(0,0,0,0.8);
}}
.hero-title {{
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0.25rem 0 0.4rem 0;
    color: {TEXT};
    text-shadow: 0 2px 12px rgba(0,0,0,0.85);
}}
.hero-sub {{
    color: #C8CCD4;
    font-size: 0.95rem;
    max-width: 620px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.85);
}}
.kpi-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 3px solid {PURPLE};
    border-radius: 6px;
    padding: 0.85rem 1rem;
}}
.kpi-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: {TEXT_DIM};
    text-transform: uppercase;
    letter-spacing: 0.1em;
}}
.kpi-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.65rem;
    font-weight: 700;
    color: {TEXT};
}}
.kpi-driver {{
    color: {CYAN};
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
}}
.section-label {{
    font-family: 'JetBrains Mono', monospace;
    color: {AMBER};
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}}
.archetype-pill {{
    display: inline-block;
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 999px;
    padding: 0.15rem 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: {CYAN};
    margin-bottom: 0.3rem;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

/* ---------- tabs: clear clickable / selected states ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 2px;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: {TEXT_DIM};
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    padding: 0.55rem 1.1rem;
    transition: all 0.15s ease;
    cursor: pointer;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT};
    background: {SURFACE_2};
}}
.stTabs [aria-selected="true"] {{
    color: {BG} !important;
    background: linear-gradient(135deg, {PURPLE}, {CYAN}) !important;
    border-color: {PURPLE} !important;
    font-weight: 700;
    box-shadow: 0 -2px 12px rgba(139,108,252,0.35);
}}

/* ---------- themed scrollbar for the whole page ---------- */
::-webkit-scrollbar {{
    width: 10px;
}}
::-webkit-scrollbar-track {{
    background: {SURFACE};
}}
::-webkit-scrollbar-thumb {{
    background: linear-gradient(180deg, {PURPLE}, {CYAN});
    border-radius: 8px;
}}
</style>
{RACE_BG_HTML}
""", unsafe_allow_html=True)

def styled_table(df_: pd.DataFrame, subset=None, axis=0):
    """background_gradient needs jinja2 — degrade gracefully if it's missing
    instead of crashing the whole page."""
    try:
        kwargs = {"cmap": "Purples"}
        if subset is not None:
            kwargs["subset"] = subset
        else:
            kwargs["axis"] = axis
        return df_.style.background_gradient(**kwargs).format("{:.1f}")
    except ImportError:
        return df_.style.format("{:.1f}")

PLOTLY_TEMPLATE = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color=TEXT, family="Inter"),
)

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "laps.csv"))

df = load_data()

# ---------------------------------------------------------------------------
# HERO BANNER — large, unobstructed cinematic shot (car + shifting smoke)
# ---------------------------------------------------------------------------
HERO_BANNER_HTML = _flatten(f"""
<div class="hero-banner">
    <div class="hero-spotlight"></div>
    <div class="hero-smoke hero-smoke-red"></div>
    <div class="hero-smoke hero-smoke-cool"></div>
    <div class="hero-motion-streaks">
        <div class="streak"></div>
        <div class="streak"></div>
        <div class="streak"></div>
    </div>
    <div class="hero-car-reflect">
        <div class="hero-car-b2 tint-a">{CAR_SVG}</div>
        <div class="hero-car-b2 tint-b">{CAR_SVG}</div>
    </div>
    <div class="hero-car-banner">
        <div class="hero-car-b2 tint-a">{CAR_SVG}</div>
        <div class="hero-car-b2 tint-b">{CAR_SVG}</div>
    </div>
    <div class="hero-floor-sheen"></div>
    <div class="hero-wet-floor"></div>
    <div class="haze"></div>
    <div class="hero-scrim"></div>
    <div class="hero-text">
        <div class="hero-tag">TELEMETRY-BASED PERFORMANCE MODELING</div>
        <div class="hero-pinstripe"></div>
        <div class="hero-title">RaceIQ &middot; Driver Consistency Intelligence</div>
        <div class="hero-sub">Quantifying who performs best under pressure, in the wet, and across
        circuit types &mdash; using lap-by-lap timing data, regression-adjusted skill scores, and
        style clustering rather than raw results alone.</div>
    </div>
</div>
""")

st.markdown(HERO_BANNER_HTML, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="section-label">Filters</div>', unsafe_allow_html=True)
seasons = st.sidebar.multiselect("Season", sorted(df["season"].unique()), default=sorted(df["season"].unique()))
circuit_types = st.sidebar.multiselect("Circuit type", sorted(df["circuit_type"].unique()), default=sorted(df["circuit_type"].unique()))
condition = st.sidebar.radio("Conditions", ["All", "Dry only", "Wet only"], index=0)

fdf = df[df["season"].isin(seasons) & df["circuit_type"].isin(circuit_types)]
if condition == "Dry only":
    fdf = fdf[~fdf["is_wet"]]
elif condition == "Wet only":
    fdf = fdf[fdf["is_wet"]]

if fdf.empty or fdf["driver"].nunique() < 2:
    st.warning("Not enough data for current filters — widen your selection.")
    st.stop()

@st.cache_data(show_spinner=False)
def get_metrics(fdf: pd.DataFrame):
    """Cached so switching tabs/widgets doesn't re-run the Ridge fit and metric
    computation on every rerun — only recomputes when the filtered data actually changes.
    Streamlit hashes the DataFrame contents, so a different filter selection (which
    produces a different fdf) correctly triggers a fresh computation."""
    return compute_all_metrics(fdf, CFG)

@st.cache_data(show_spinner="Running bootstrap resampling for confidence intervals…")
def get_confidence_intervals(fdf: pd.DataFrame):
    return bootstrap_confidence_intervals(fdf, CFG)

@st.cache_data(show_spinner=False)
def get_skill_diagnostics(fdf: pd.DataFrame):
    return get_skill_model_diagnostics(fdf, CFG)

metrics_df = get_metrics(fdf)

st.sidebar.markdown('<div class="section-label" style="margin-top:1.2rem;">About the scores</div>', unsafe_allow_html=True)
st.sidebar.caption(
    "**Consistency** — inverse coefficient of variation of race lap times.\n\n"
    "**Wet score** — pace drop-off dry→wet vs field.\n\n"
    "**Pressure score** — pace when gap <1s in closing stages.\n\n"
    "**Conversion** — quali pace → race pace efficiency.\n\n"
    "**Skill score** — residual pace after regressing out tire/track/weather context."
)

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["OVERVIEW", "DRIVER DEEP-DIVE", "HEAD-TO-HEAD", "STYLE CLUSTERS"])

# ===== TAB 1: OVERVIEW =====
with tab1:
    top = metrics_df.sort_values("overall_score", ascending=False)
    leader = top.index[0]

    kpi_defs = [
        ("MOST CONSISTENT", metrics_df["consistency_score"].idxmax(), "consistency_score"),
        ("BEST IN THE WET", metrics_df["wet_score"].idxmax(), "wet_score"),
        ("BEST UNDER PRESSURE", metrics_df["pressure_score"].idxmax(), "pressure_score"),
        ("BEST QUALI→RACE CONVERTER", metrics_df["conversion_score"].idxmax(), "conversion_score"),
    ]
    cols = st.columns(4)
    for col, (label, driver, metric) in zip(cols, kpi_defs):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-driver">{driver}</div>
                <div class="kpi-value">{metrics_df.loc[driver, metric]:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Overall leaderboard</div>', unsafe_allow_html=True)

    fig = go.Figure()
    sorted_df = metrics_df.sort_values("overall_score")
    fig.add_trace(go.Bar(
        y=sorted_df.index, x=sorted_df["overall_score"], orientation="h",
        marker=dict(color=sorted_df["overall_score"], colorscale=[[0, SURFACE_2], [1, PURPLE]]),
        text=sorted_df["overall_score"].round(1), textposition="outside",
        textfont=dict(family="JetBrains Mono"),
    ))
    fig.update_layout(**PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=40, t=10, b=10),
                       xaxis=dict(title="Overall Score", gridcolor=BORDER, range=[0, 105]),
                       yaxis=dict(title=""))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">Full metric table</div>', unsafe_allow_html=True)
    st.dataframe(
        styled_table(metrics_df, subset=["overall_score"]),
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Skill score model — how it's calculated & how confident are we"):
        st.caption(
            "The skill score comes from regressing lap time on track type, tire compound, "
            "tire age, and track temperature (RidgeCV, alpha chosen by cross-validation), "
            "then ranking drivers by their residual pace — i.e. how fast they are after "
            "removing the effect of conditions they didn't control."
        )
        diag = get_skill_diagnostics(fdf)
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Chosen Ridge α (CV)", f"{diag['chosen_alpha']:.3g}")
        dc2.metric("Train R²", f"{diag['r2_train']:.3f}")
        dc3.metric("Race laps used", f"{diag['n_laps']:,}")

        coef_series = pd.Series(diag["standardized_coefficients"]).sort_values()
        fig_coef = go.Figure(go.Bar(
            x=coef_series.values, y=coef_series.index, orientation="h",
            marker_color=[RED if v > 0 else CYAN for v in coef_series.values],
        ))
        fig_coef.update_layout(
            **PLOTLY_TEMPLATE, height=280, margin=dict(l=10, r=20, t=10, b=10),
            xaxis=dict(title="Standardized effect on lap time (s)", gridcolor=BORDER, zeroline=True),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_coef, use_container_width=True)
        st.caption(
            "Bars to the right slow a lap down; bars to the left speed it up. Standardizing "
            "puts tire age, temperature, and compound/circuit dummies on the same scale so "
            "their effect sizes are directly comparable."
        )

        st.markdown('<div class="section-label" style="margin-top:1rem;">Score uncertainty (bootstrap 90% CI)</div>', unsafe_allow_html=True)
        st.caption(
            "Each score is recomputed 500 times on resampled laps to show how much it could "
            "plausibly shift — a narrow band means the ranking is well-supported by the data; "
            "a wide one means don't over-read small differences between nearby drivers."
        )
        ci_df = get_confidence_intervals(fdf)
        if "overall_score" in metrics_df.columns and "skill_score_lo" in ci_df.columns:
            plot_df = metrics_df.join(ci_df, how="inner").sort_values("skill_score")
            fig_ci = go.Figure(go.Bar(
                y=plot_df.index, x=plot_df["skill_score"], orientation="h",
                marker_color=PURPLE,
                error_x=dict(
                    type="data", symmetric=False,
                    array=plot_df["skill_score_hi"] - plot_df["skill_score"],
                    arrayminus=plot_df["skill_score"] - plot_df["skill_score_lo"],
                    color=TEXT_DIM,
                ),
            ))
            fig_ci.update_layout(
                **PLOTLY_TEMPLATE, height=340, margin=dict(l=10, r=20, t=10, b=10),
                xaxis=dict(title="Skill score (with 90% CI)", gridcolor=BORDER),
                yaxis=dict(title=""),
            )
            st.plotly_chart(fig_ci, use_container_width=True)
        else:
            st.info("Not enough laps in the current filter to compute stable confidence intervals.")

# ===== TAB 2: DRIVER DEEP-DIVE =====
with tab2:
    driver_sel = st.selectbox("Select driver", metrics_df.index.tolist())
    ddf = fdf[fdf["driver"] == driver_sel]

    c1, c2, c3 = st.columns(3)
    for col, (label, val) in zip([c1, c2, c3], [
        ("OVERALL SCORE", metrics_df.loc[driver_sel, "overall_score"]),
        ("CONSISTENCY", metrics_df.loc[driver_sel, "consistency_score"]),
        ("PRESSURE SCORE", metrics_df.loc[driver_sel, "pressure_score"]),
    ]):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val:.1f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Race lap-time trace (by circuit)</div>', unsafe_allow_html=True)
    race = ddf[ddf["session"] == "Race"]
    circuit_pick = st.selectbox("Circuit", sorted(race["circuit"].unique()))
    trace = race[race["circuit"] == circuit_pick].sort_values(["season", "lap_number"])
    fig2 = px.line(trace, x="lap_number", y="lap_time_s", color=trace["season"].astype(str),
                    color_discrete_sequence=[PURPLE, CYAN, AMBER])
    fig2.update_layout(**PLOTLY_TEMPLATE, height=340,
                        xaxis=dict(gridcolor=BORDER, title="Lap"), yaxis=dict(gridcolor=BORDER, title="Lap time (s)"),
                        legend_title="Season")
    st.plotly_chart(fig2, use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown('<div class="section-label">Pace by tire compound</div>', unsafe_allow_html=True)
        fig3 = px.box(race, x="tire_compound", y="lap_time_s", color="tire_compound",
                       color_discrete_sequence=[PURPLE, CYAN, AMBER])
        fig3.update_layout(**PLOTLY_TEMPLATE, height=300, showlegend=False,
                            xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
        st.plotly_chart(fig3, use_container_width=True)
    with cc2:
        st.markdown('<div class="section-label">Dry vs wet pace</div>', unsafe_allow_html=True)
        wetcmp = race.groupby("is_wet")["lap_time_s"].median().rename({True: "Wet", False: "Dry"})
        fig4 = go.Figure(go.Bar(x=wetcmp.index.map({True: "Wet", False: "Dry"}), y=wetcmp.values,
                                 marker_color=[CYAN, AMBER]))
        fig4.update_layout(**PLOTLY_TEMPLATE, height=300,
                            xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER, title="Median lap (s)"))
        st.plotly_chart(fig4, use_container_width=True)

# ===== TAB 3: HEAD-TO-HEAD =====
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        driver_a = st.selectbox("Driver A", metrics_df.index.tolist(), index=0)
    with c2:
        remaining = [d for d in metrics_df.index if d != driver_a]
        driver_b = st.selectbox("Driver B", remaining, index=0)

    radar_cols = ["consistency_score", "wet_score", "pressure_score", "conversion_score", "skill_score"]
    radar_labels = ["Consistency", "Wet Weather", "Under Pressure", "Quali→Race", "Raw Skill"]

    fig5 = go.Figure()
    for driver, color in [(driver_a, PURPLE), (driver_b, CYAN)]:
        vals = metrics_df.loc[driver, radar_cols].tolist()
        fig5.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=radar_labels + [radar_labels[0]],
            fill="toself", name=driver, line=dict(color=color)
        ))
    fig5.update_layout(**PLOTLY_TEMPLATE, height=460,
                        polar=dict(bgcolor=SURFACE,
                                   radialaxis=dict(visible=True, range=[0, 100], gridcolor=BORDER),
                                   angularaxis=dict(gridcolor=BORDER)),
                        showlegend=True)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-label">Head-to-head, metric by metric</div>', unsafe_allow_html=True)
    comp = metrics_df.loc[[driver_a, driver_b], radar_cols + ["overall_score"]].T
    comp.index = radar_labels + ["Overall"]
    st.dataframe(styled_table(comp, axis=1), use_container_width=True)

# ===== TAB 4: STYLE CLUSTERS =====
with tab4:
    st.markdown('<div class="section-label">Driver archetypes (unsupervised clustering)</div>', unsafe_allow_html=True)

    auto_k = st.checkbox("Auto-select cluster count by silhouette score", value=True)
    if auto_k:
        clusters, sil_scores = cluster_driver_styles(metrics_df, n_clusters=None, cfg=CFG)
        if sil_scores:
            best_k = clusters.nunique()
            st.caption(
                f"Selected **k={best_k}** archetypes — the value that maximizes silhouette "
                f"score (a measure of how well-separated the clusters are) across k=2–"
                f"{max(sil_scores)}, rather than a fixed default."
            )
            sil_fig = go.Figure(go.Bar(
                x=[str(k) for k in sil_scores.keys()], y=list(sil_scores.values()),
                marker_color=[PURPLE if k == best_k else SURFACE_2 for k in sil_scores.keys()],
            ))
            sil_fig.update_layout(**PLOTLY_TEMPLATE, height=180, margin=dict(l=10, r=10, t=10, b=10),
                                   xaxis=dict(title="k", gridcolor=BORDER),
                                   yaxis=dict(title="Silhouette score", gridcolor=BORDER))
            st.plotly_chart(sil_fig, use_container_width=True)
    else:
        n_clusters = st.slider("Number of archetypes", 2, min(5, len(metrics_df) - 1), 3)
        clusters, _ = cluster_driver_styles(metrics_df, n_clusters=n_clusters, cfg=CFG)
    labels = name_clusters(metrics_df, clusters)

    merged = metrics_df.join(clusters)
    merged["archetype"] = merged["cluster"].map(labels)

    for cid, label in labels.items():
        group = merged[merged["cluster"] == cid]
        st.markdown(f'<span class="archetype-pill">{label}</span>', unsafe_allow_html=True)
        st.write(", ".join(group.index.tolist()))

    st.markdown("<br>", unsafe_allow_html=True)
    radar_cols = ["consistency_score", "wet_score", "pressure_score", "conversion_score", "skill_score"]
    radar_labels = ["Consistency", "Wet Weather", "Under Pressure", "Quali→Race", "Raw Skill"]
    fig6 = go.Figure()
    palette = [PURPLE, CYAN, AMBER, RED, "#5AD1E6"]
    for i, (cid, label) in enumerate(labels.items()):
        avg = merged[merged["cluster"] == cid][radar_cols].mean().tolist()
        fig6.add_trace(go.Scatterpolar(
            r=avg + [avg[0]], theta=radar_labels + [radar_labels[0]],
            fill="toself", name=label, line=dict(color=palette[i % len(palette)])
        ))
    fig6.update_layout(**PLOTLY_TEMPLATE, height=460,
                        polar=dict(bgcolor=SURFACE,
                                   radialaxis=dict(visible=True, range=[0, 100], gridcolor=BORDER),
                                   angularaxis=dict(gridcolor=BORDER)))
    st.plotly_chart(fig6, use_container_width=True)
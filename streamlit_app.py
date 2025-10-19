import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_theme import st_theme

# -------------------------------
# Utilities
# -------------------------------

# Create exactly ONE component instance with a unique key:
_APP_THEME = st_theme(key="app_theme_component") or {}

def legend_theme_kwargs():
    """
    Legend colors that follow the active Streamlit theme.
    Falls back to Streamlit's config if the component returns None initially.
    """
    base = (_APP_THEME.get("base") or (st.get_option("theme.base") or "light")).lower()
    is_dark = (base == "dark")

    bg   = _APP_THEME.get("secondaryBackgroundColor") or st.get_option("theme.secondaryBackgroundColor") or ("#0E1117" if is_dark else "#FFFFFF")
    text = _APP_THEME.get("textColor")                 or st.get_option("theme.textColor")                 or ("#FAFAFA" if is_dark else "#31333F")
    border = "#404040" if is_dark else "#E0E0E0"
    return dict(fillColor=bg, strokeColor=border, labelColor=text, titleColor=text)

def layer_label_color() -> str:
    base = (_APP_THEME.get("base") or (st.get_option("theme.base") or "light")).lower()
    is_dark = (base == "dark")
    # white on dark theme, near-black on light theme
    return "#FFFFFF" if is_dark else (st.get_option("theme.textColor") or "#31333F")


def dewpoint_c(T_c: float, RH_pct: float) -> float:
    """Dewpoint based on Magnus/Tetens (approx.)."""
    RH = max(1e-6, min(RH_pct, 100.0)) / 100.0
    a, b = 17.27, 237.3
    alpha = ((a * T_c) / (b + T_c)) + math.log(RH)
    return (b * alpha) / (a - alpha)


# -------------------------------
# Utilities
# -------------------------------

@st.cache_data
def default_layers_df() -> pd.DataFrame:
    """A neutral, illustrative wall build-up.
    You can overwrite in the data editor.
    Columns:
      - Name
      - Thickness (m)
      - Lambda (W/mK)
      - mu (–)  (vapour diffusion resistance factor)
    """
    data = [
        {"Name": "Gypsum board", "Thickness (m)": 0.0125, "Lambda (W/mK)": 0.25, "mu (–)": 10.0},
        {"Name": "Mineral wool", "Thickness (m)": 0.10,   "Lambda (W/mK)": 0.040, "mu (–)": 1.0},
        {"Name": "OSB",          "Thickness (m)": 0.012,  "Lambda (W/mK)": 0.13,  "mu (–)": 50.0},
        {"Name": "Brick",        "Thickness (m)": 0.10,   "Lambda (W/mK)": 0.77,  "mu (–)": 10.0},
    ]
    df = pd.DataFrame(data)
    return df


def sat_vapor_pressure_pa(T_c: float) -> float:
    """Saturation vapor pressure over liquid water (Pa) using Tetens for T >= 0°C
    and over ice for T < 0°C. See Monteith & Unsworth (2008) / Murray (1967).
    """
    if T_c >= 0.0:
        # Tetens over water: kPa -> Pa
        return 0.61078 * math.exp((17.27 * T_c) / (T_c + 237.3)) * 1000.0
    else:
        # Tetens over ice: kPa -> Pa
        return 0.61078 * math.exp((21.875 * T_c) / (T_c + 265.5)) * 1000.0


def pv_from_rh(T_c: float, rh_pct: float) -> float:
    return sat_vapor_pressure_pa(T_c) * max(0.0, min(rh_pct, 100.0)) / 100.0




@dataclass
class Layer:
    name: str
    thickness_m: float
    lambda_w_mk: float
    mu: float

    @property
    def R(self) -> float:
        if self.lambda_w_mk <= 0 or self.thickness_m <= 0:
            return 0.0
        return self.thickness_m / self.lambda_w_mk

    @property
    def sd(self) -> float:
        # Equivalent air layer thickness (m) for vapor diffusion
        return max(0.0, self.mu) * max(0.0, self.thickness_m)


# -------------------------------
# Core Glaser calculation
# -------------------------------

def glaser(
    layers: List[Layer],
    T_int_c: float,
    RH_int_pct: float,
    T_ext_c: float,
    RH_ext_pct: float,
    R_si: float = 0.13,
    R_se: float = 0.04,
    sd_si: float = 0.0,
    sd_se: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (nodes_table, layers_table) for visualization.

    nodes_table has one row per interface (including interior & exterior surfaces), with:
    x_m, cum_R, T_c, p_sat_Pa, p_v_Pa, condenses

    layers_table summarizes layer-by-layer metrics (R, sd, etc.).
    """
    # Boundary vapor pressures from indoor/outdoor RH
    pvi = pv_from_rh(T_int_c, RH_int_pct)
    pve = pv_from_rh(T_ext_c, RH_ext_pct)

    # Thermal resistances and positions
    R_layers = np.array([max(0.0, L.thickness_m) / max(L.lambda_w_mk, 1e-12) for L in layers], dtype=float)
    x_layers = np.array([max(0.0, L.thickness_m) for L in layers], dtype=float)
    sd_layers = np.array([L.sd for L in layers], dtype=float)

    R_total = R_si + R_layers.sum() + R_se
    # positions at interfaces (including the two surfaces)
    x_interfaces = np.concatenate([[0.0], np.cumsum(x_layers)])

    # cumulative thermal resistance at interfaces (start with R_si at interior surface)
    cum_R = np.concatenate([[0.0], np.cumsum(R_layers)]) + R_si
    # add external surface resistance at the end for total only (not needed point-wise)

    # Linear temperature profile vs cumulative R
    delta_T = T_int_c - T_ext_c
    T_interfaces = T_int_c - (cum_R / R_total) * delta_T

    # Vapor resistance path (include optional surface sd)
    sd_total = sd_si + sd_layers.sum() + sd_se
    sd_cum = sd_si + np.concatenate([[0.0], np.cumsum(sd_layers)])

    # Linear vapor pressure profile along sd (from inside to outside)
    p_v_interfaces = pvi - (sd_cum / max(sd_total, 1e-12)) * (pvi - pve)

    # Saturation vapor pressure at interface temperatures
    p_sat_interfaces = np.array([sat_vapor_pressure_pa(T) for T in T_interfaces])

    condenses = p_v_interfaces > p_sat_interfaces

    nodes = pd.DataFrame(
        {
            "Position x (m)": x_interfaces,
            "Cum. R (m²K/W)": cum_R,
            "Temperature (°C)": T_interfaces,
            "p_sat (Pa)": p_sat_interfaces,
            "p_v (Pa)": p_v_interfaces,
            "Condensation?": condenses,
        }
    )

    layers_tbl = pd.DataFrame(
        {
            "Layer": [L.name for L in layers],
            "Thickness (m)": x_layers,
            "Lambda (W/mK)": [L.lambda_w_mk for L in layers],
            "mu (–)": [L.mu for L in layers],
            "R (m²K/W)": R_layers,
            "sd (m)": sd_layers,
        }
    )

    return nodes, layers_tbl




# -------------------------------
# UI
# -------------------------------

st.set_page_config(page_title="Glaser Method – Condensation Check")

st.title("Glaser Method – Interstitial Condensation")

st.markdown(
    """
    This app implements the **Glaser method** (EN ISO 13788) for checking interstitial condensation risk in a
    1D multi-layer building element under **steady-state** conditions. Provide the indoor/outdoor
    boundary conditions and layer data (thickness, thermal conductivity, and vapour diffusion
    resistance factor μ). The app computes:

    * temperature profile
    * saturation vapour pressure profile
    * actual vapour pressure profile (from vapour diffusion)

    and flags locations where $p_v$ > $p_{sat}$ (risk of condensation). Use with care for design guidance;
    it is not a hygrothermal simulation (no storage, capillarity, or transient effects).
    """
)

with st.sidebar:
    st.header("Boundary conditions")
    colA, colB = st.columns(2)
    with colA:
        T_int = st.number_input("Indoor temperature (°C)", value=20.0, step=0.5, format="%.1f")
        RH_int = st.number_input("Indoor RH (%)", value=50.0, step=1.0, min_value=0.0, max_value=100.0)
        R_si = st.number_input("Rsi (m²K/W)", value=0.13, step=0.01)
        sd_si = st.number_input("sd (inside) (m)", value=0.0, step=0.01, help="Optional surface vapour resistance as equivalent air layer thickness")
    with colB:
        T_ext = st.number_input("Outdoor temperature (°C)", value=0.0, step=0.5, format="%.1f")
        RH_ext = st.number_input("Outdoor RH (%)", value=80.0, step=1.0, min_value=0.0, max_value=100.0)
        R_se = st.number_input("Rse (m²K/W)", value=0.04, step=0.01)
        sd_se = st.number_input("sd (outside) (m)", value=0.0, step=0.01)

    st.divider()
    st.caption("Preset just populates the table below – you can fully edit afterwards.")
    if st.button("Load illustrative preset layers"):
        st.session_state["layers_df"] = default_layers_df()

    # Plot overlays
    show_layers = st.checkbox(
        "Show material boundaries & labels",
        value=True,
        help="Draw vertical dashed lines at each layer boundary and show layer names"
    )

# Data editor for layers
st.subheader("Layers (from inside → outside)")

if "layers_df" not in st.session_state:
    st.session_state["layers_df"] = default_layers_df()

edited_df = st.data_editor(
    st.session_state["layers_df"],
    num_rows="dynamic",
    width='stretch',
    column_config={
        "Name": st.column_config.TextColumn(required=True),
        "Thickness (m)": st.column_config.NumberColumn(min_value=0.0, step=0.001, format="%.4f"),
        "Lambda (W/mK)": st.column_config.NumberColumn(min_value=0.0, step=0.001, format="%.3f"),
        "mu (–)": st.column_config.NumberColumn(min_value=0.0, step=0.5, format="%.1f"),
    },
)

# Convert edited rows to Layer objects (skip empty rows)
layers: List[Layer] = []
for _, row in edited_df.iterrows():
    try:
        name = str(row.get("Name", "Layer"))
        d = float(row.get("Thickness (m)", 0) or 0)
        lam = float(row.get("Lambda (W/mK)", 0) or 0)
        mu = float(row.get("mu (–)", 0) or 0)
    except Exception:
        continue
    if d > 0 and lam > 0:
        layers.append(Layer(name=name, thickness_m=d, lambda_w_mk=lam, mu=mu))

if len(layers) == 0:
    st.info("Add at least one layer with positive thickness and thermal conductivity.")
    st.stop()

# Run Glaser (unchanged)
nodes, layers_tbl = glaser(layers, T_int, RH_int, T_ext, RH_ext, R_si, R_se, sd_si, sd_se)

# Build layer bounds for overlays (unchanged)
bounds = []
x0 = 0.0
for L in layers:
    x1 = x0 + L.thickness_m
    bounds.append({"Layer": L.name, "x0": x0, "x1": x1, "xc": 0.5 * (x0 + x1)})
    x0 = x1
bands_df = pd.DataFrame(bounds)
# ALL interfaces (inside, between layers, outside) from computed nodes:
edges_df = pd.DataFrame({"x": np.unique(nodes["Position x (m)"].values)})

# =========================
# Temperature profile (TOP)
# =========================
st.markdown("### Temperature profile")

# Add a constant series label so the chart shows a legend
temp_plot_df = nodes.assign(Series="Temperature (°C)")

temp_legend = alt.Legend(
    title="Legend",
    orient="top-right",
    cornerRadius=6,
    padding=6,
    **legend_theme_kwargs()
)

base_temp = (
    alt.Chart(temp_plot_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("Position x (m):Q", title="Depth (m)"),
        y=alt.Y("Temperature (°C):Q", title="Temperature (°C)"),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(domain=["Temperature (°C)"], range=["#d62728"]),
            legend=temp_legend
        )
    )
)

# THEME-DEPENDENT LAYER LABEL COLOR (white on dark, dark on light)
label_color = layer_label_color()  

if show_layers and len(bands_df):
    rules = alt.Chart(edges_df).mark_rule(
        strokeDash=[6, 3], opacity=0.8, strokeWidth=1.5, color=label_color,   
    ).encode(x="x:Q")

    labels = alt.Chart(bands_df).mark_text(
        dy=-30,
        opacity=0.95,
        color=label_color,      
        # optional readability outline:
        # stroke="black", strokeWidth=2, strokeOpacity=0.25
    ).encode(
        x="xc:Q", y=alt.value(14), text="Layer:N"
    )

    temp_chart = base_temp + rules + labels
else:
    temp_chart = base_temp

st.altair_chart(
    temp_chart.properties(
        # width=900,
        height=300
    )
)

# ==========================
# Vapour pressures (BOTTOM)
# ==========================
st.markdown("### Vapour pressures")

# Long form for two lines
plot_df = nodes.melt(
    id_vars=["Position x (m)"],
    value_vars=["p_sat (Pa)", "p_v (Pa)"],
    var_name="Series",
    value_name="Pa"
)

press_legend = alt.Legend(
    title="Legend",
    orient="top-right",
    cornerRadius=6,
    padding=6,
    **legend_theme_kwargs()
)

base_press = (
    alt.Chart(plot_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("Position x (m):Q", title="Depth (m)"),
        y=alt.Y("Pa:Q", title="Pressure (Pa)"),
        color=alt.Color("Series:N", legend=press_legend)
    )
)

if show_layers and len(bands_df):
    rules2 = alt.Chart(edges_df).mark_rule(
        strokeDash=[6, 3], opacity=0.8, strokeWidth=1.5, color=label_color,  
    ).encode(x="x:Q")
    press_chart = base_press + rules2 + labels  # reuses the same labels if you want them here too
else:
    press_chart = base_press

st.altair_chart(
    press_chart.properties(
        # width=863,
        height=300
    )
)



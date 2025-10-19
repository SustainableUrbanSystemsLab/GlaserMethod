import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_theme import st_theme
st.set_page_config(layout="wide")

# Inject custom CSS to set the width of the sidebar
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            min-width: 300px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    base = (_APP_THEME.get("base") or (
        st.get_option("theme.base") or "light")).lower()
    is_dark = (base == "dark")

    bg = _APP_THEME.get("secondaryBackgroundColor") or st.get_option(
        "theme.secondaryBackgroundColor") or ("#0E1117" if is_dark else "#FFFFFF")
    text = _APP_THEME.get("textColor") or st.get_option(
        "theme.textColor") or ("#FAFAFA" if is_dark else "#31333F")
    border = "#404040" if is_dark else "#E0E0E0"
    return dict(fillColor=bg, strokeColor=border, labelColor=text, titleColor=text)


def layer_label_color() -> str:
    base = (_APP_THEME.get("base") or (
        st.get_option("theme.base") or "light")).lower()
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
      - μ (–)  (vapour diffusion resistance factor)
    """
    data = [
        {"Name": "Plaster", "Thickness (m)": 0.02,
         "Lambda (W/mK)": 0.35, "μ (–)": 10},
        {"Name": "Limestone",
            "Thickness (m)": 0.24,   "Lambda (W/mK)": 1.4, "μ (–)": 48},
        {"Name": "EPS032",
            "Thickness (m)": 0.12,  "Lambda (W/mK)": 0.032,  "μ (–)": 20},
        {"Name": "Lime Cem. Plaster",
            "Thickness (m)": 0.02,   "Lambda (W/mK)": 1.0,  "μ (–)": 35},
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
    R_layers = np.array([max(0.0, L.thickness_m) /
                        max(L.lambda_w_mk, 1e-12) for L in layers], dtype=float)
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
    p_sat_interfaces = np.array([sat_vapor_pressure_pa(T)
                                for T in T_interfaces])

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
            "μ (–)": [L.mu for L in layers],
            "R (m²K/W)": R_layers,
            "sd (m)": sd_layers,
        }
    )

    return nodes, layers_tbl


# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Glaser Method – Condensation Check")

st.title("Glaser Method")

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
        T_int = st.number_input("Indoor temperature (°C)",
                                value=20.0, step=0.5, format="%.1f")
        RH_int = st.number_input(
            "Indoor RH (%)", value=50.0, step=1.0, min_value=0.0, max_value=100.0)
        R_si = st.number_input("R_si (m²K/W)", value=0.25, step=0.01)
        sd_si = st.number_input("sd (inside) (m)", value=0.0, step=0.01,
                                help="Optional surface vapour resistance as equivalent air layer thickness")
    with colB:
        T_ext = st.number_input(
            "Outdoor temperature (°C)", value=-5.0, step=0.5, format="%.1f")
        RH_ext = st.number_input(
            "Outdoor RH (%)", value=80.0, step=1.0, min_value=0.0, max_value=100.0)
        R_se = st.number_input("R_se (m²K/W)", value=0.04, step=0.01)
        sd_se = st.number_input("sd (outside) (m)", value=0.0, step=0.01)

    st.divider()
    st.caption(
        "Preset just populates the table on the right – you can fully edit afterwards.")
    if st.button("Load illustrative preset layers"):
        st.session_state["layers_df"] = default_layers_df()

    # Plot overlays
    show_layers = st.checkbox(
        "Show material boundaries & labels",
        value=True,
        help="Draw vertical dashed lines at each layer boundary and show layer names"
    )

st.divider()

# Data editor for layers
st.subheader("Input Layers (from inside → outside)")

if "layers_df" not in st.session_state:
    st.session_state["layers_df"] = default_layers_df()

edited_df = st.data_editor(
    st.session_state["layers_df"],
    num_rows="dynamic",
    width='stretch',
    column_config={
        "Name": st.column_config.TextColumn(required=True),
        "Thickness (m)": st.column_config.NumberColumn(min_value=0.0, step=0.001, format="%.2f"),
        "Lambda (W/mK)": st.column_config.NumberColumn(min_value=0.0, step=0.001, format="%.2f"),
        "μ (–)": st.column_config.NumberColumn(min_value=0.0, step=0.5, format="%.0f"),
    },
)

# Convert edited rows to Layer objects (skip empty rows)
layers: List[Layer] = []
for _, row in edited_df.iterrows():
    try:
        name = str(row.get("Name", "Layer"))
        d = float(row.get("Thickness (m)", 0) or 0)
        lam = float(row.get("Lambda (W/mK)", 0) or 0)
        mu = float(row.get("μ (–)", 0) or 0)
    except Exception:
        continue
    if d > 0 and lam > 0:
        layers.append(Layer(name=name, thickness_m=d, lambda_w_mk=lam, mu=mu))

if len(layers) == 0:
    st.info("Add at least one layer with positive thickness and thermal conductivity.")
    st.stop()

# Run Glaser
nodes, layers_tbl = glaser(layers, T_int, RH_int,
                           T_ext, RH_ext, R_si, R_se, sd_si, sd_se)

# Build layer bounds for overlays
bounds = []
x0 = 0.0
for L in layers:
    x1 = x0 + L.thickness_m
    bounds.append({"Layer": L.name, "x0": x0, "x1": x1, "xc": 0.5 * (x0 + x1)})
    x0 = x1
bands_df = pd.DataFrame(bounds)
# ALL interfaces (inside, between layers, outside) from computed nodes:
edges_df = pd.DataFrame({"x": np.unique(nodes["Position x (m)"].values)})


# --- Condensation by layer (use p_v - p_sat at segment endpoints; linear => endpoints suffice) ---
diff_ps = (nodes["p_v (Pa)"].to_numpy() -
           nodes["p_sat (Pa)"].to_numpy())  # length = n_layers + 1
cond_by_layer = []
for i in range(len(layers)):
    d0, d1 = diff_ps[i], diff_ps[i+1]
    # any positive or a zero-crossing within layer
    cond = (d0 > 0) or (d1 > 0) or (d0 * d1 < 0)
    cond_by_layer.append(cond)

# Label text per layer: append a warning symbol if that layer has condensation risk
layer_texts = [f"{L.name} ⚠" if c else L.name for L,
               c in zip(layers, cond_by_layer)]


st.divider()

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
# Vapour pressure vs cumulative sd
# ==========================
# st.markdown("### Vapour pressure vs. cumulative sd")

# --- cumulative sd boundaries (length n+3) ---
sd_layers = layers_tbl["sd (m)"].to_numpy(dtype=float)          # n
sd_after = sd_si + np.cumsum(sd_layers)                         # n
sd_total = float(sd_si + sd_layers.sum() + sd_se)
# 0, sd_si, after each layer, sd_total
cum_sd = np.concatenate(([0.0], [sd_si], sd_after, [sd_total]))


# --- names matching those boundaries (length n+3) ---
n = len(layers)
names = ["InternalAir"]
if n >= 1:
    names.append(f"InternalAir_{layers[0].name}")
    for i in range(n - 1):
        names.append(f"{layers[i].name}_{layers[i+1].name}")
    names.append(f"{layers[-1].name}_ExternalAir")
names.append("ExternalAir")

# --- temperatures at those boundaries (length n+3) ---
temps = [float(T_int)] + \
    nodes["Temperature (°C)"].astype(float).tolist() + [float(T_ext)]

# --- vapor pressures at those boundaries (length n+3) ---
# p_v (Glaser line) endpoints from indoor/outdoor RH
pvi = pv_from_rh(T_int, RH_int)
pve = pv_from_rh(T_ext, RH_ext)
# n+1 (interior surface + after each layer)
sd_interfaces = np.concatenate(([sd_si], sd_after))
p_v_interfaces = pvi - (sd_interfaces / max(sd_total, 1e-12)) * (pvi - pve)
p_v_full = [float(pvi)] + \
    p_v_interfaces.astype(float).tolist() + [float(pve)]  # n+3
# p_sat(T) at each boundary
p_sat_full = [sat_vapor_pressure_pa(t) for t in temps]  # n+3

# --- bring "Condensation?" into this table/label ---

# nodes["Condensation?"] is defined at each interface (interior surface + after each layer)
# Our sd-axis table has boundaries: [InternalAir] + interfaces + [ExternalAir]
# length n+3
cond_flags = [False] + nodes["Condensation?"].astype(bool).tolist() + [False]

# Add a '⚠' to the boundary label when there's condensation at that boundary
label_with_flag = [f"{nm} ⚠" if flag else nm for nm,
                   flag in zip(names, cond_flags)]

# --- assemble the table (with flagged label and a boolean column) ---
layers_T_wVapor = pd.DataFrame({
    # label now includes ⚠ when needed
    "Layer_Boundary": label_with_flag,
    "Temperature_C":  np.round(temps, 2),
    "p_v_Pa":         np.round(p_v_full).astype(int),
    "p_sat_Pa":       np.round(p_sat_full).astype(int),
    "cum_sd":         np.round(cum_sd, 2),
    # keep the plain boolean too
    "Condensation?":  cond_flags,
})


# ==========================
# Vapor pressure vs. cumulative sd
# ==========================
st.markdown("### Vapor pressure vs. cumulative sd")

# Toggle
show_phys = st.checkbox(
    "Show physical p_v line",
    value=True,
    help="Toggle the physical Glaser result (piecewise line constrained by p_sat)."
)

# Use your table
s_all = layers_T_wVapor["cum_sd"].to_numpy(
    float)       # [0, sd_si, ..., sd_total]
pv_all = layers_T_wVapor["p_v_Pa"].to_numpy(float)       # endpoints: pvi, pve
ps_all = layers_T_wVapor["p_sat_Pa"].to_numpy(float)

# Interfaces where Glaser checks saturation (interior surface and after each layer)
s_ifc = s_all[1:-1]                                     # length n+1
ps_ifc = ps_all[1:-1]                                    # length n+1
# internal air boundary
sL, pL = 0.0, pv_all[0]
# external air boundary
sR, pR = float(s_all[-1]), float(pv_all[-1])

# --- iterative pull-down (only computed if needed) ---
knot_s, knot_p = [sL], [pL]
if show_phys:
    start_idx = 0
    while True:
        denom = (sR - sL) or 1e-12
        line_at_ifc = pL + (pR - pL) * (s_ifc[start_idx:] - sL) / denom
        above = line_at_ifc > ps_ifc[start_idx:]
        if not np.any(above):
            break
        i_rel = int(np.argmax(above))   # first violation
        hit = start_idx + i_rel
        sL, pL = float(s_ifc[hit]), float(ps_ifc[hit])  # snap to saturation
        knot_s.append(sL)
        knot_p.append(pL)
        start_idx = hit + 1
    knot_s.append(sR)
    knot_p.append(pR)

# Data for plotting
df_pv = layers_T_wVapor[["cum_sd", "p_v_Pa"]].rename(
    columns={"p_v_Pa": "Pa"}).assign(Series="No-condensation p_v")
df_ps = layers_T_wVapor[["cum_sd", "p_sat_Pa"]].rename(
    columns={"p_sat_Pa": "Pa"}).assign(Series="Saturation p_sat")

dfs = [df_ps, df_pv]
series_domain = ["Saturation p_sat", "No-condensation p_v"]
series_range = ["gray", "steelblue"]

if show_phys:
    df_phys = pd.DataFrame(
        {"cum_sd": knot_s, "Pa": knot_p, "Series": "Physical p_v"})
    # interior contact points
    df_knots = pd.DataFrame({"cum_sd": knot_s[1:-1], "Pa": knot_p[1:-1]})
    dfs.append(df_phys)
    series_domain.append("Physical p_v")
    series_range.append("purple")
else:
    df_knots = pd.DataFrame(columns=["cum_sd", "Pa"])

series_df = pd.concat(dfs, ignore_index=True)

legend = alt.Legend(title="Legend", orient="top-right",
                    cornerRadius=6, padding=6, **legend_theme_kwargs())

rules = (
    alt.Chart(layers_T_wVapor)
    .mark_rule(strokeDash=[6, 3], opacity=0.8, strokeWidth=1.5, color=label_color)
    .encode(x="cum_sd:Q")
)

chart_phys = (
    alt.Chart(series_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("cum_sd:Q", title="Cumulative sd (m)"),
        y=alt.Y("Pa:Q", title="Water Vapor Pressure (Pa)"),
        color=alt.Color("Series:N", legend=legend,
                        scale=alt.Scale(domain=series_domain, range=series_range)),
    )
)

# if you add knot markers:
knots_mark = alt.Chart(df_knots).mark_point(
    color="crimson", size=80).encode(x="cum_sd:Q", y="Pa:Q")


# red markers at condensation "knot" points (only when enabled and there are interior knots)
final_phys = chart_phys + rules
if show_phys and "knot_s" in locals() and len(knot_s) > 2:
    knots_df = pd.DataFrame({"cum_sd": knot_s[1:-1], "Pa": knot_p[1:-1]})
    knots_mark = alt.Chart(knots_df).mark_point(color="crimson", size=80).encode(
        x="cum_sd:Q", y="Pa:Q"
    )
    final_phys = final_phys + knots_mark  # add layer only in this case

st.altair_chart(final_phys.properties(height=300), use_container_width=True)


# =========================
# Results & tables (BOTTOM)
# =========================

st.divider()

st.markdown("### Layer summary")
col1, col2, col3, col4, col5 = st.columns(5, gap='Small', border=True)
with col1:
    st.metric("Wall thickness (m)", f"{layers_tbl['Thickness (m)'].sum():.2f}")
with col2:
    st.metric("R_Total (m²K/W)",
              f"{layers_tbl['R (m²K/W)'].sum() + R_si + R_se:.2f}")
with col3:
    st.metric("Heat flux density q (W/m2)",
              f"{1/(layers_tbl['R (m²K/W)'].sum() + R_si + R_se)* abs(T_ext-T_int) :.1f}")
with col4:
    st.metric("Any condensation?",
              "Yes" if nodes["Condensation?"].any() else "No")
with col5:
    st.metric("Indoor dewpoint (°C)", f"{dewpoint_c(T_int, RH_int):.2f}")


st.dataframe(layers_T_wVapor)
st.caption("**Note:** “⚠” marks a boundary where p_v > p_sat at that node.")


# Downloads
csv_nodes = nodes.to_csv(index=False).encode("utf-8")
csv_layers = layers_tbl.to_csv(index=False).encode("utf-8")
colA, colB = st.columns(2)
with colA:
    st.download_button("Download interface results (CSV)", data=csv_nodes,
                       file_name="glaser_interfaces.csv", mime="text/csv")
with colB:
    st.download_button("Download layer summary (CSV)", data=csv_layers,
                       file_name="glaser_layers.csv", mime="text/csv")


st.divider()

# Notes
st.info(
    """
    **Notes**
    - Method follows the classic steady-state *Glaser* approach: linear temperature and vapour pressure
      profiles based on thermal and vapour resistances.
    - Surface vapour resistances (sd) are optional and default to 0 m.
    - For rigorous design on moisture safety, prefer transient hygrothermal simulation tools
      (e.g., EN 15026 / WUFI) in addition to Glaser.
    """
)

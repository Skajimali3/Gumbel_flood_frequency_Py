#Gumbel Flood Frequency Analysis in Python 3.11

This repository provides Python 3.11 scripts for flood frequency analysis using the Gumbel Extreme Value Type-I distribution. The workflow was developed to estimate transboundary flood frequency, peak discharge probability, and return periods for the Kosi and Teesta river basins.

"""
=============================================================================
Transboundary Flood Frequency Analysis Using Gumbel's Extreme Value
Distribution (EV-I) — TRASH-FI Framework
=============================================================================
Manuscript: "Upstream–Downstream Flood Dynamics and Socio-Hydrological
            Inequalities in Transboundary River Basins"

Description:
    This script implements the complete Gumbel EV-I flood frequency analysis
    for upstream and downstream gauging stations in the Kosi and Teesta
    transboundary river basins. It includes:
        1. Parameter estimation (Method of Moments)
        2. Return period quantile estimation
        3. Weibull plotting position (empirical probabilities)
        4. Goodness-of-fit (Kolmogorov–Smirnov test)
        5. 95% confidence intervals (Kite, 1977)
        6. Gumbel probability paper plots
        7. Summary statistics table export

Dependencies:
    pip install numpy pandas scipy matplotlib seaborn openpyxl
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import kstest, gumbel_r
from scipy.special import gamma
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
EULER_MASCHERONI = 0.5772156649   # γ (Euler–Mascheroni constant)
RETURN_PERIODS   = [2, 5, 10, 25, 50, 100, 200, 500]  # years
ALPHA_CI         = 0.05           # significance level for K–S test
OUTPUT_DIR       = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. STATION METADATA
#    Replace the synthetic discharge series with your observed AMS data.
#    Format: {station_id: {"name": str, "basin": str, "reach": str,
#                          "ams": list/array of annual maximum discharge (m³/s)}}
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)

STATIONS = {
    # ── KOSI BASIN ── UPSTREAM ──────────────────────────────────────────────
    "KFU1": {
        "name":  "Bahraibise",
        "basin": "Kosi",
        "reach": "Upstream",
        "ams":   np.random.gumbel(loc=60,    scale=20,   size=50).clip(min=5),
    },
    "KFU2": {
        "name":  "Thame",
        "basin": "Kosi",
        "reach": "Upstream",
        "ams":   np.random.gumbel(loc=1500,  scale=400,  size=50).clip(min=100),
    },
    "KFU3": {
        "name":  "Kushaha",
        "basin": "Kosi",
        "reach": "Upstream",
        "ams":   np.random.gumbel(loc=55,    scale=18,   size=50).clip(min=5),
    },
    # ── KOSI BASIN ── DOWNSTREAM ─────────────────────────────────────────────
    "KFU4": {
        "name":  "Laukahi",
        "basin": "Kosi",
        "reach": "Downstream",
        "ams":   np.random.gumbel(loc=18000, scale=3000, size=50).clip(min=1000),
    },
    "KFU5": {
        "name":  "Hanumannagar",
        "basin": "Kosi",
        "reach": "Downstream",
        "ams":   np.random.gumbel(loc=1200,  scale=350,  size=50).clip(min=100),
    },
    "KFD1": {
        "name":  "Nirmali",
        "basin": "Kosi",
        "reach": "Downstream",
        "ams":   np.random.gumbel(loc=20000, scale=4000, size=50).clip(min=2000),
    },
    "KFD2": {
        "name":  "Bangaon",
        "basin": "Kosi",
        "reach": "Downstream",
        "ams":   np.random.gumbel(loc=15000, scale=3500, size=50).clip(min=1000),
    },
    "KFD3": {
        "name":  "Alam Nagar",
        "basin": "Kosi",
        "reach": "Downstream",
        "ams":   np.random.gumbel(loc=12000, scale=2500, size=50).clip(min=800),
    },
    "KFD4": {
        "name":  "Kursela",
        "basin": "Kosi",
        "reach": "Downstream",
        "ams":   np.random.gumbel(loc=1000,  scale=250,  size=50).clip(min=100),
    },
    # ── TEESTA BASIN ── UPSTREAM ─────────────────────────────────────────────
    "TFU1": {
        "name":  "Singhik",
        "basin": "Teesta",
        "reach": "Upstream",
        "ams":   np.random.gumbel(loc=90,    scale=30,   size=50).clip(min=10),
    },
    "TFU2": {
        "name":  "Singtam",
        "basin": "Teesta",
        "reach": "Upstream",
        "ams":   np.random.gumbel(loc=1500,  scale=450,  size=50).clip(min=100),
    },
    # ── TEESTA BASIN ── MID/DOWNSTREAM ───────────────────────────────────────
    "TFU3": {
        "name":  "Malbazar",
        "basin": "Teesta",
        "reach": "Downstream",
        "ams":   np.random.gumbel(loc=9000,  scale=1800, size=50).clip(min=500),
    },
    "TFU4": {
        "name":  "Dakhin Nijtaraf",
        "basin": "Teesta",
        "reach": "Downstream",
        "ams":   np.random.gumbel(loc=10500, scale=2000, size=50).clip(min=500),
    },
    "TFD1": {
        "name":  "Hatibandha-1",
        "basin": "Teesta",
        "reach": "Downstream",
        "ams":   np.random.gumbel(loc=40,    scale=15,   size=50).clip(min=2),
    },
    "TFD2": {
        "name":  "Hatibandha-2",
        "basin": "Teesta",
        "reach": "Downstream",
        "ams":   np.random.gumbel(loc=55,    scale=20,   size=50).clip(min=2),
    },
    "TFD3": {
        "name":  "Hatibandha-3",
        "basin": "Teesta",
        "reach": "Downstream",
        "ams":   np.random.gumbel(loc=70,    scale=25,   size=50).clip(min=2),
    },
    "TFD4": {
        "name":  "Gaibandha-1",
        "basin": "Teesta",
        "reach": "Downstream",
        "ams":   np.random.gumbel(loc=35,    scale=20,   size=50).clip(min=2),
    },
    "TFD5": {
        "name":  "Gaibandha-2",
        "basin": "Teesta",
        "reach": "Downstream",
        "ams":   np.random.gumbel(loc=80,    scale=30,   size=50).clip(min=2),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. CORE GUMBEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def gumbel_parameters(ams: np.ndarray) -> tuple:
    """
    Estimate Gumbel EV-I parameters using Method of Moments (MOM).

    Parameters
    ----------
    ams : array-like — Annual Maximum Series (m³/s)

    Returns
    -------
    alpha : float — scale parameter  (Eq. 2)
    u     : float — location parameter / mode (Eq. 3)
    x_bar : float — sample mean
    s     : float — sample standard deviation
    n     : int   — record length
    """
    ams   = np.asarray(ams, dtype=float)
    n     = len(ams)
    x_bar = np.mean(ams)                          # Eq. 4
    s     = np.std(ams, ddof=1)                   # Eq. 5
    alpha = (np.sqrt(6) / np.pi) * s             # Eq. 2
    u     = x_bar - EULER_MASCHERONI * alpha      # Eq. 3
    return alpha, u, x_bar, s, n


def reduced_variate(T: float) -> float:
    """
    Gumbel reduced variate y_T for return period T (years).   (Eq. 7)
    """
    return -np.log(-np.log(1.0 - 1.0 / T))


def frequency_factor(T: float) -> float:
    """
    Gumbel frequency factor K_T.                               (Eq. 9)
    """
    y = reduced_variate(T)
    return (np.sqrt(6) / np.pi) * (-EULER_MASCHERONI - np.log(-np.log(1 - 1/T)))


def quantile_discharge(T: float, x_bar: float, s: float) -> float:
    """
    Discharge quantile x_T for return period T.               (Eq. 8)
    """
    KT = frequency_factor(T)
    return x_bar + KT * s


def confidence_interval(T: float, x_bar: float, s: float, n: int,
                         alpha_ci: float = 0.05) -> tuple:
    """
    95% confidence interval on Gumbel quantile estimate.
    Standard error from Kite (1977).                          (Eqs. 13–14)
    """
    KT  = frequency_factor(T)
    SE  = (s / np.sqrt(n)) * np.sqrt(1 + 1.1396 * KT + 1.1 * KT**2)
    xT  = quantile_discharge(T, x_bar, s)
    z   = 1.96                                    # 95% CI
    return xT - z * SE, xT + z * SE


def weibull_plotting_position(ams: np.ndarray) -> tuple:
    """
    Weibull plotting position formula.                        (Eqs. 10–11)

    Returns
    -------
    x_sorted   : sorted AMS (descending)
    T_empirical: empirical return period
    y_empirical: empirical reduced variate
    """
    ams_sorted = np.sort(ams)[::-1]               # descending
    n          = len(ams_sorted)
    m          = np.arange(1, n + 1)
    T_emp      = (n + 1) / m                      # Eq. 11
    p_exc      = m / (n + 1)                      # Eq. 10
    y_emp      = -np.log(-np.log(1 - p_exc))      # reduced variate
    return ams_sorted, T_emp, y_emp


def exceedance_probability(Q_c: float, alpha: float, u: float) -> float:
    """
    Exceedance probability for threshold discharge Q_c.       (Eq. 15)
    """
    return 1.0 - np.exp(-np.exp(-(Q_c - u) / alpha))


def transboundary_flood_return_period(Q_c: float, alpha: float, u: float) -> float:
    """
    Transboundary Flood Return Period (TFRP) for Q_c.         (Eq. 16)
    """
    p = exceedance_probability(Q_c, alpha, u)
    return 1.0 / p if p > 0 else np.inf


def ks_goodness_of_fit(ams: np.ndarray, alpha: float, u: float) -> tuple:
    """
    Kolmogorov–Smirnov test against fitted Gumbel CDF.        (Eq. 12)

    Returns: (D_n, p_value, reject_H0)
    """
    # Standardise using fitted parameters
    standardised = (ams - u) / alpha
    D, p_val     = kstest(standardised, "gumbel_r")
    reject       = p_val < ALPHA_CI
    return D, p_val, reject

# ─────────────────────────────────────────────────────────────────────────────
# 5. ANALYSIS LOOP — compute results for all stations
# ─────────────────────────────────────────────────────────────────────────────

results = []

for sid, meta in STATIONS.items():
    ams   = np.asarray(meta["ams"], dtype=float)
    alpha, u, x_bar, s, n = gumbel_parameters(ams)
    D, p_ks, reject        = ks_goodness_of_fit(ams, alpha, u)

    row = {
        "Station ID": sid,
        "Station Name": meta["name"],
        "Basin":        meta["basin"],
        "Reach":        meta["reach"],
        "N (years)":    n,
        "Mean (m³/s)":  round(x_bar, 2),
        "Std Dev":      round(s, 2),
        "α (scale)":    round(alpha, 2),
        "u (location)": round(u, 2),
        "K-S Stat":     round(D, 4),
        "K-S p-value":  round(p_ks, 4),
        "Reject H₀ (5%)": reject,
    }

    for T in RETURN_PERIODS:
        xT       = quantile_discharge(T, x_bar, s)
        lb, ub   = confidence_interval(T, x_bar, s, n)
        row[f"Q_{T}yr (m³/s)"]    = round(xT, 1)
        row[f"Q_{T}yr CI_lower"]  = round(lb, 1)
        row[f"Q_{T}yr CI_upper"]  = round(ub, 1)

    results.append(row)

df_results = pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────────────────
# 6. EXPORT SUMMARY TABLE TO EXCEL
# ─────────────────────────────────────────────────────────────────────────────

excel_path = os.path.join(OUTPUT_DIR, "Gumbel_TFF_Results.xlsx")

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_results.to_excel(writer, sheet_name="All Stations", index=False)

    for basin in ["Kosi", "Teesta"]:
        df_b = df_results[df_results["Basin"] == basin]
        df_b.to_excel(writer, sheet_name=basin, index=False)

print(f"✔  Results exported → {excel_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATION — Gumbel Probability Paper Plots (Fig. 4 style)
# ─────────────────────────────────────────────────────────────────────────────

# ── Colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    "Upstream":   "#2166ac",   # blue
    "Downstream": "#d6604d",   # red-orange
}

# ── Reduced variate axis ticks ────────────────────────────────────────────────
T_plot  = np.logspace(np.log10(1.01), np.log10(1000), 300)
y_plot  = reduced_variate(T_plot)
T_ticks = [2, 5, 10, 25, 50, 100, 200, 500]
y_ticks = [reduced_variate(t) for t in T_ticks]


def plot_gumbel_station(ax, sid: str, meta: dict, show_xlabel: bool = True,
                         show_ylabel: bool = True):
    """Plot Gumbel probability paper for a single station."""
    ams             = np.asarray(meta["ams"], dtype=float)
    alpha, u, x_bar, s, n = gumbel_parameters(ams)
    colour          = COLOURS[meta["reach"]]

    # Empirical plotting positions
    x_obs, T_emp, y_emp = weibull_plotting_position(ams)

    # Theoretical line
    x_fit = x_bar + frequency_factor_array(y_plot, s, x_bar)
    lb_fit, ub_fit = [], []
    for T in T_plot:
        lb, ub = confidence_interval(T, x_bar, s, n)
        lb_fit.append(lb)
        ub_fit.append(ub)

    # Plot
    ax.scatter(y_emp, x_obs, color=colour, s=20, zorder=5,
               label="Observed (Weibull)", edgecolors="white", linewidths=0.4)
    ax.plot(y_plot, x_fit, color=colour, lw=1.8, label="Gumbel EV-I fit")
    ax.fill_between(y_plot, lb_fit, ub_fit, color=colour, alpha=0.15,
                    label="95% CI")

    # Axes formatting
    ax.set_xticks(y_ticks)
    ax.set_xticklabels([str(t) for t in T_ticks], fontsize=7)
    ax.set_xlim(y_ticks[0] - 0.2, y_ticks[-1] + 0.3)
    ax.tick_params(axis="y", labelsize=7)

    if show_xlabel:
        ax.set_xlabel("Return Period (years)", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Discharge (m³ s⁻¹)", fontsize=8)

    # Annotation
    D, p_ks, _ = ks_goodness_of_fit(ams, alpha, u)
    ax.set_title(f"{sid} — {meta['name']}\n({meta['basin']}, {meta['reach']})",
                 fontsize=8, fontweight="bold", pad=3)
    ax.annotate(f"α={alpha:.1f}  u={u:.1f}\nK–S p={p_ks:.3f}  n={n}",
                xy=(0.03, 0.97), xycoords="axes fraction",
                fontsize=6.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def frequency_factor_array(y_array, s, x_bar):
    """Vectorised quantile calculation from reduced variate array."""
    KT_array = (np.sqrt(6) / np.pi) * (-EULER_MASCHERONI - y_array)
    return x_bar + KT_array * s


# ── Main figure: all 18 stations ─────────────────────────────────────────────
station_ids = list(STATIONS.keys())
n_stations  = len(station_ids)
ncols       = 3
nrows       = int(np.ceil(n_stations / ncols))

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(15, nrows * 3.5),
                          constrained_layout=True)
axes_flat = axes.flatten()

for idx, sid in enumerate(station_ids):
    ax = axes_flat[idx]
    show_x = (idx >= n_stations - ncols)
    show_y = (idx % ncols == 0)
    plot_gumbel_station(ax, sid, STATIONS[sid],
                        show_xlabel=show_x, show_ylabel=show_y)

# Hide unused subplots
for idx in range(n_stations, len(axes_flat)):
    axes_flat[idx].set_visible(False)

# Common legend
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac",
           markersize=6, label="Upstream — Observed"),
    Line2D([0], [0], color="#2166ac", lw=1.8, label="Upstream — Gumbel fit"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#d6604d",
           markersize=6, label="Downstream — Observed"),
    Line2D([0], [0], color="#d6604d", lw=1.8, label="Downstream — Gumbel fit"),
]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=4, fontsize=8, frameon=True,
           bbox_to_anchor=(0.5, -0.015))

fig.suptitle(
    "Transboundary Flood Frequency Analysis — Gumbel EV-I Distribution\n"
    "Kosi and Teesta River Basins (18 Gauging Stations)",
    fontsize=11, fontweight="bold", y=1.01
)

plot_path = os.path.join(OUTPUT_DIR, "Fig4_Gumbel_TFF_All_Stations.png")
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✔  Fig. 4 saved → {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. UPSTREAM vs DOWNSTREAM COMPARISON PLOT (Fig. 4 summary panel)
# ─────────────────────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

for ax, basin in zip(axes2, ["Kosi", "Teesta"]):
    basin_stations = {k: v for k, v in STATIONS.items() if v["basin"] == basin}
    T_range        = np.logspace(np.log10(1.01), np.log10(500), 300)
    y_range        = reduced_variate(T_range)

    for sid, meta in basin_stations.items():
        ams              = np.asarray(meta["ams"], dtype=float)
        _, _, x_bar, s, n = gumbel_parameters(ams)
        colour           = COLOURS[meta["reach"]]
        lw               = 1.2
        x_fit            = frequency_factor_array(y_range, s, x_bar)
        ax.plot(T_range, x_fit, color=colour, lw=lw, alpha=0.75,
                label=f"{sid} — {meta['name']}")

    ax.set_xscale("log")
    ax.set_xticks(T_ticks)
    ax.set_xticklabels([str(t) for t in T_ticks])
    ax.set_xlabel("Return Period (years)", fontsize=10)
    ax.set_ylabel("Discharge (m³ s⁻¹)", fontsize=10)
    ax.set_title(f"{basin} Basin — Upstream vs Downstream\nGumbel EV-I Flood Frequency",
                 fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Custom legend patches
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=COLOURS["Upstream"],   label="Upstream stations"),
        Patch(color=COLOURS["Downstream"], label="Downstream stations"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper left", frameon=True)

plot_path2 = os.path.join(OUTPUT_DIR, "Fig4_Gumbel_Basin_Comparison.png")
fig2.savefig(plot_path2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"✔  Basin comparison plot saved → {plot_path2}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. TRANSBOUNDARY FLOOD RETURN PERIOD (TFRP) TABLE
#    Define a critical threshold discharge Q_c per station
#    (e.g. bankfull or warning level — adjust as required)
# ─────────────────────────────────────────────────────────────────────────────

tfrp_records = []

for sid, meta in STATIONS.items():
    ams             = np.asarray(meta["ams"], dtype=float)
    alpha, u, x_bar, s, n = gumbel_parameters(ams)
    Q_c             = np.percentile(ams, 90)       # 90th percentile as Q_c
    p_exc           = exceedance_probability(Q_c, alpha, u)
    tfrp            = transboundary_flood_return_period(Q_c, alpha, u)

    tfrp_records.append({
        "Station ID":         sid,
        "Station Name":       meta["name"],
        "Basin":              meta["basin"],
        "Reach":              meta["reach"],
        "Q_c (m³/s) — 90th": round(Q_c, 2),
        "Exceedance Prob.":   round(p_exc, 4),
        "TFRP (years)":       round(tfrp, 1),
    })

df_tfrp = pd.DataFrame(tfrp_records)
print("\n── Transboundary Flood Return Period (TFRP) Summary ────────────────")
print(df_tfrp.to_string(index=False))

tfrp_path = os.path.join(OUTPUT_DIR, "TFRP_Summary.xlsx")
df_tfrp.to_excel(tfrp_path, index=False)
print(f"\n✔  TFRP table exported → {tfrp_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. PRINT FULL RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Gumbel EV-I Parameters & Return Period Quantiles ────────────────")
display_cols = (
    ["Station ID", "Station Name", "Basin", "Reach", "N (years)",
     "Mean (m³/s)", "Std Dev", "α (scale)", "u (location)",
     "K-S p-value", "Reject H₀ (5%)"]
    + [f"Q_{T}yr (m³/s)" for T in RETURN_PERIODS]
)
print(df_results[display_cols].to_string(index=False))
print("\n✔  Analysis complete. All outputs saved to:", OUTPUT_DIR)

----------------Python code for Flood Frequency Analysis-------------------------------

"""
=============================================================================
Transboundary Flood Frequency Analysis Using Gumbel's Extreme Value
Distribution (EV-I)
=============================================================================

Code
"""
=============================================================================
Transboundary Flood Frequency Analysis using Gumbel's Extreme Value
Distribution (EV-I): Teesta and Kosi basin reaches
=============================================================================
This script reproduces, step by step, the calculation in the workbook
"Gumbel_FFA_Teesta_Kosi_100y_RP.xlsx" (sheets Method, Annual_Max,
station sheets and Summary_QT).
 

Data
Annual maximum daily discharge (AMS, m3/s) for 18 reaches,
    calendar years 1940 to 2025 (N = 86). 
    The AMS is read from sheet "Annual_Max" of the workbook. As an option,
    the AMS can be rebuilt from the daily retrospective CSV files.
 
Steps (same as each station sheet in the workbook)
    1. Annual maximum series (AMS).
    2. Rank AMS in descending order, m = 1 ... N.
    3. Weibull plotting position: P = m/(N+1); T = (N+1)/m.
    4. Reduced variate of each ranked flood: y = -ln(-ln(1 - P)).
    5. Sample mean (x_bar) and standard deviation (sigma, n-1) of the AMS.
    6. Reduced mean y_n = mean of the N reduced variates;
       reduced SD S_n = population SD of the N reduced variates
       (same as Gumbel y_n / S_n tables; N = 86: y_n = 0.558, S_n = 1.198).
    7. For each design return period T:
           y_T = -ln(-ln((T-1)/T))
           K   = (y_T - y_n) / S_n          (finite-sample frequency factor)
           Q_T = x_bar + K * sigma
    8. 95% confidence limits (Kite, 1977):
           b  = sqrt(1 + 1.1396 K + 1.1 K^2)
           Se = b * sigma / sqrt(N)
           Q_T +/- 1.96 Se
    9. For comparison, Q_T with the large-sample (Chow) factor
           K_inf = -(sqrt(6)/pi) * [0.5772 + ln(ln(T/(T-1)))]
   10. Fitted Gumbel line at each observed point:
           Q = x_bar + sigma * (y - y_n) / S_n
       Gumbel parameters: alpha (scale) = sigma/S_n ; u (location) = x_bar - alpha*y_n
 
    Kolmogorov-Smirnov goodness of fit and the Transboundary Flood Return
    Period (TFRP) table, both using the alpha and u above.
 
Dependencies
    pip install numpy pandas scipy matplotlib openpyxl
=============================================================================
"""
 
# -----------------------------------------------------------------------------
# 1. IMPORTS
# -----------------------------------------------------------------------------
import os
import glob
import warnings
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import kstest, skew
 
warnings.filterwarnings("ignore")
 
# -----------------------------------------------------------------------------
# 2. CONFIGURATION
# -----------------------------------------------------------------------------
WORKBOOK       = "Gumbel_FFA_Teesta_Kosi_100y_RP.xlsx"   # input workbook
AMS_SOURCE     = "excel"        # "excel" = sheet Annual_Max ; "csv" = daily CSV files
CSV_FOLDER     = "daily_csv"    # used only when AMS_SOURCE = "csv"
FIRST_YEAR     = 1940
LAST_YEAR      = 2025           # 2026 incomplete, excluded
 
RETURN_PERIODS = [2, 5, 10, 25, 50, 100]   # years (as in the workbook)
EULER_CHOW     = 0.5772                    # constant used in Chow K_inf (workbook value)
Z_95           = 1.96                      # 95% confidence
ALPHA_KS       = 0.05                      # significance level for K-S test
 
OUTPUT_DIR     = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# -----------------------------------------------------------------------------
# 3. STATION LIST (sheet "Stations")
#    Order and basin as in the workbook. "reach" (Upstream/Downstream) is used
# -----------------------------------------------------------------------------
STATIONS = [
    # name,              basin,    reach ID,    reach,        CSV file name
    ("Singhik",          "Teesta", "441131477", "Upstream",   "U_Teesta_retrospective_441131477_Singhik.csv"),
    ("Singtam",          "Teesta", "441146631", "Upstream",   "Teesta_Singtam_retrospective_441146631.csv"),
    ("Malbazar",         "Teesta", "441104000", " Upstream", "Teesta_Malbazar_retrospective_441104000.csv"),
    ("Dakhin Nijtaraf",  "Teesta", "441095755", " Upstream", "Teesta_Dakhin_Nijtaraf_retrospective_441095755.csv"),
    ("Patgram",          "Teesta", "441156325", "Downstream", "retrospective_441156325_Patgram.csv"),
    ("Hatibandha",       "Teesta", "441101294", "Downstream", "retrospective_441101294_Hatibandha.csv"),
    ("Aditmari",         "Teesta", "441104046", "Downstream", "retrospective_441104046_Aditmari.csv"),
    ("Chilmari",         "Teesta", "441190744", "Downstream", "retrospective_441190744_chilmari.csv"),
    ("Gaibandha",        "Teesta", "441183870", "Downstream", "retrospective_441183870_gaibandha.csv"),
    ("Bahrabise",        "Kosi",   "441183778", "Upstream",   "retrospective_441183778_Bahrabise.csv"),
    ("Thame",            "Kosi",   "441164508", "Upstream",   "retrospective_441164508_Thame.csv"),
    ("Kushaha",          "Kosi",   "441175538", "Upstream",   "retrospective_441175538_Kushaha.csv"),
    ("Hanumannagar",     "Kosi",   "441167306", " Upstream", "retrospective_441167306_Hanumannagar.csv"),
    ("Laukahi",          "Kosi",   "441138414", " Upstream", "retrospective_441138414_Laukahi.csv"),
    ("Nirmali",          "Kosi",   "441099900", "Downstream", "retrospective_441099900_Nirmali.csv"),
    ("Bangaon",          "Kosi",   "441154960", "Downstream", "retrospective_441154960_Bangaon.csv"),
    ("Alamnagar",        "Kosi",   "441201752", "Downstream", "retrospective_441201752_Alamnagar.csv"),
    ("Kursela",          "Kosi",   "441124706", "Downstream", "retrospective_441124706_Kursela.csv"),
]
 
# -----------------------------------------------------------------------------
# 4. READ ANNUAL MAXIMUM SERIES
# -----------------------------------------------------------------------------
 
def read_ams_from_excel(path):
    """
    Read sheet Annual_Max: row 3 = header (Year, station columns),
    rows 4 onward = one row per year. Returns AMS table and peak dates.
    """
    ams = pd.read_excel(path, sheet_name="Annual_Max", header=2)
    ams = ams.rename(columns={ams.columns[0]: "Year"})
    ams["Year"] = pd.to_numeric(ams["Year"], errors="coerce")
    ams = ams.dropna(subset=["Year"])
    ams["Year"] = ams["Year"].astype(int)
    ams = ams[(ams["Year"] >= FIRST_YEAR) & (ams["Year"] <= LAST_YEAR)]
 
    dates = pd.read_excel(path, sheet_name="Peak_Dates", header=2)
    dates = dates.rename(columns={dates.columns[0]: "Year"})
    dates["Year"] = pd.to_numeric(dates["Year"], errors="coerce")
    dates = dates.dropna(subset=["Year"])
    dates["Year"] = dates["Year"].astype(int)
 
    # column headers are "Name\n(reach ID)"; rename them to the station name
    def clean(df):
        new = {}
        for col in df.columns[1:]:
            new[col] = str(col).split("\n")[0].strip()
        return df.rename(columns=new)
 
    return clean(ams).set_index("Year"), clean(dates).set_index("Year")
 
 
def read_ams_from_csv(folder):
    """
    Optional: build the AMS from the daily retrospective CSV files.
    Assumes each CSV has a date/time column and a discharge column.
    Change DATE_COL and Q_COL below to match your files.
    Dates are rounded to the nearest day (fixes the 23:00 UTC shift
    in Oct 1941 to 1945). Calendar year annual maximum is used.
    """
    DATE_COL = "time"        # change to your date column name
    Q_COL    = "discharge"   # change to your discharge column name
 
    ams_table   = {}
    dates_table = {}
    for name, basin, rid, reach, fname in STATIONS:
        f = os.path.join(folder, fname)
        if not os.path.exists(f):
            hits = glob.glob(os.path.join(folder, f"*{rid}*.csv"))
            if not hits:
                raise FileNotFoundError(f"No CSV found for {name} ({rid})")
            f = hits[0]
        d = pd.read_csv(f)
        d[DATE_COL] = pd.to_datetime(d[DATE_COL]).dt.round("D")
        d["Year"] = d[DATE_COL].dt.year
        d = d[(d["Year"] >= FIRST_YEAR) & (d["Year"] <= LAST_YEAR)]
        idx = d.groupby("Year")[Q_COL].idxmax()
        ams_table[name]   = d.loc[idx].set_index("Year")[Q_COL].round(2)
        dates_table[name] = d.loc[idx].set_index("Year")[DATE_COL].dt.strftime("%Y-%m-%d")
    return pd.DataFrame(ams_table), pd.DataFrame(dates_table)
 
 
if AMS_SOURCE == "excel":
    AMS, PEAK_DATES = read_ams_from_excel(WORKBOOK)
else:
    AMS, PEAK_DATES = read_ams_from_csv(CSV_FOLDER)
 
print(f"AMS loaded: {AMS.shape[1]} stations, {AMS.shape[0]} years "
      f"({AMS.index.min()} to {AMS.index.max()})")
 
# -----------------------------------------------------------------------------
# 5. CORE GUMBEL FUNCTIONS (same formulas as the station sheets)
# -----------------------------------------------------------------------------
 
def reduced_variate_T(T):
    """y_T = -ln(-ln((T-1)/T))   (design return period T)"""
    T = np.asarray(T, dtype=float)
    return -np.log(-np.log((T - 1.0) / T))
 
 
def ranked_table(years, q):
    """
    Steps 2 to 4: rank AMS (descending), Weibull P and T, reduced variate y.
    Year of each ranked flood is the first year with that value
    (same as INDEX/MATCH in the workbook).
    """
    q     = np.asarray(q, dtype=float)
    years = np.asarray(years)
    n     = len(q)
    order = np.argsort(-q, kind="stable")      # descending, first year kept for ties
    m     = np.arange(1, n + 1)
    P     = m / (n + 1.0)                      # Weibull plotting position
    T     = (n + 1.0) / m                      # return period
    y     = -np.log(-np.log(1.0 - P))          # reduced variate
    return pd.DataFrame({
        "Rank m": m,
        "Year": years[order],
        "Q observed, ranked (m³/s)": q[order],
        "P = m/(N+1)": P,
        "T = (N+1)/m (yr)": T,
        "Reduced variate y": y,
    })
 
 
def gumbel_parameters(q, y):
    """
    Steps 5 and 6: sample statistics, reduced mean and SD, Gumbel alpha and u.
    sigma uses n-1 (STDEV); S_n uses population SD (STDEVP) of y.
    """
    q = np.asarray(q, dtype=float)
    p = {}
    p["N"]     = len(q)
    p["x_bar"] = np.mean(q)
    p["sigma"] = np.std(q, ddof=1)
    p["y_n"]   = np.mean(y)
    p["S_n"]   = np.std(y, ddof=0)
    p["Cv"]    = p["sigma"] / p["x_bar"]
    p["Cs"]    = skew(q, bias=False)          # same as Excel SKEW
    p["Q_max"] = np.max(q)
    p["Q_min"] = np.min(q)
    p["alpha"] = p["sigma"] / p["S_n"]         # Gumbel scale
    p["u"]     = p["x_bar"] - p["alpha"] * p["y_n"]   # Gumbel location
    return p
 
 
def design_floods(p, return_periods=RETURN_PERIODS):
    """
    Steps 7 to 9: y_T, K, Q_T, Kite 95% limits, Chow K_inf and Q_T(K_inf).
    """
    rows = []
    for T in return_periods:
        yT    = reduced_variate_T(T)
        K     = (yT - p["y_n"]) / p["S_n"]
        QT    = p["x_bar"] + K * p["sigma"]
        b     = np.sqrt(1 + 1.1396 * K + 1.1 * K ** 2)
        Se    = b * p["sigma"] / np.sqrt(p["N"])
        K_inf = -np.sqrt(6) / np.pi * (EULER_CHOW + np.log(np.log(T / (T - 1.0))))
        rows.append({
            "T (yr)": T,
            "Annual exceedance prob. 1/T": 1.0 / T,
            "yT": yT,
            "K = (yT-ȳn)/Sn": K,
            "QT = x̄+Kσ (m³/s)": QT,
            "b": b,
            "Se (m³/s)": Se,
            "95% lower (m³/s)": QT - Z_95 * Se,
            "95% upper (m³/s)": QT + Z_95 * Se,
            "K∞ (Chow)": K_inf,
            "QT with K∞ (m³/s)": p["x_bar"] + K_inf * p["sigma"],
        })
    return pd.DataFrame(rows)
 
 
def fitted_q(y, p):
    """Step 10: fitted Gumbel line Q = x_bar + sigma (y - y_n) / S_n"""
    return p["x_bar"] + p["sigma"] * (np.asarray(y) - p["y_n"]) / p["S_n"]
 
 
def ci_band(y, p):
    """95% band along the fitted line, using K at each y (for plots)."""
    K  = (np.asarray(y) - p["y_n"]) / p["S_n"]
    Se = np.sqrt(1 + 1.1396 * K + 1.1 * K ** 2) * p["sigma"] / np.sqrt(p["N"])
    Q  = p["x_bar"] + K * p["sigma"]
    return Q - Z_95 * Se, Q + Z_95 * Se
 
 
def ks_goodness_of_fit(q, p):
    """K-S test of the AMS against the fitted Gumbel (alpha, u)."""
    z = (np.asarray(q) - p["u"]) / p["alpha"]
    D, p_val = kstest(z, "gumbel_r")
    return D, p_val, p_val < ALPHA_KS
 
 
def exceedance_probability(Q_c, p):
    """P(Q > Q_c) = 1 - exp(-exp(-(Q_c - u)/alpha))"""
    return 1.0 - np.exp(-np.exp(-(Q_c - p["u"]) / p["alpha"]))
 
# -----------------------------------------------------------------------------
# 6. ANALYSIS LOOP: all stations
# -----------------------------------------------------------------------------
RESULTS = {}        # station name -> dict(ranked, params, design)
summary_rows = []
stations_rows = []
 
for i, (name, basin, rid, reach, fname) in enumerate(STATIONS, start=1):
    q_series = AMS[name].dropna()
    years    = q_series.index.values
    q        = q_series.values.astype(float)
 
    ranked = ranked_table(years, q)
    p      = gumbel_parameters(q, ranked["Reduced variate y"].values)
    ranked["Q estimated, Gumbel (m³/s)"] = fitted_q(ranked["Reduced variate y"], p)
    design = design_floods(p)
    D, p_ks, reject = ks_goodness_of_fit(q, p)
    p["year_of_max"] = int(ranked["Year"].iloc[0])
 
    RESULTS[name] = {"basin": basin, "rid": rid, "reach": reach,
                     "ranked": ranked, "params": p, "design": design,
                     "ks": (D, p_ks, reject)}
 
    stations_rows.append({
        "No.": i, "Station": name, "Basin": basin, "Reach ID": rid,
        "Source file": fname, "Period": f"{years.min()}–{years.max()}",
        "N (years)": p["N"], "Mean AMS (m³/s)": p["x_bar"],
        "SD AMS (m³/s)": p["sigma"], "Cv": p["Cv"],
        "Max AMS (m³/s)": p["Q_max"], "Year of max": p["year_of_max"],
    })
 
    row = {"Station": name, "Basin": basin, "Reach ID": rid,
           "x̄": p["x_bar"], "σ": p["sigma"], "ȳn": p["y_n"], "Sn": p["S_n"]}
    for _, d in design.iterrows():
        row[f"Q{int(d['T (yr)'])}"] = d["QT = x̄+Kσ (m³/s)"]
    row.update({"α (scale)": p["alpha"], "u (location)": p["u"],
                "K-S D": D, "K-S p-value": p_ks, "Reject H0 (5%)": reject})
    summary_rows.append(row)
 
df_stations = pd.DataFrame(stations_rows)
df_summary  = pd.DataFrame(summary_rows)
 
# -----------------------------------------------------------------------------
# 7. CHECK AGAINST THE WORKBOOK (Summary_QT cached values)
# -----------------------------------------------------------------------------
 
def check_against_workbook(path, df):
    try:
        ref = pd.read_excel(path, sheet_name="Summary_QT", header=3)
    except Exception as e:
        print("Workbook check skipped:", e)
        return
    ref = ref.dropna(subset=["Station"]).set_index("Station")
    cols = [f"Q{T}" for T in RETURN_PERIODS] + ["x̄", "σ", "ȳn", "Sn"]
    mine = df.set_index("Station")
    diffs = []
    for c in cols:
        if c in ref.columns:
            a = pd.to_numeric(ref[c], errors="coerce")
            b = mine.loc[a.index, c]
            diffs.append(np.nanmax(np.abs(a - b) / np.abs(a)))
    if diffs and np.all(np.isfinite(diffs)):
        print(f"Check against workbook Summary_QT: max relative difference = {max(diffs):.2e}")
    else:
        print("Workbook check: cached values not available (open and save the workbook in Excel).")
 
if AMS_SOURCE == "excel":
    check_against_workbook(WORKBOOK, df_summary)
 
# -----------------------------------------------------------------------------
# 8. EXPORT TABLES TO EXCEL
# -----------------------------------------------------------------------------
excel_path = os.path.join(OUTPUT_DIR, "Gumbel_FFA_Results.xlsx")
 
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_stations.to_excel(writer, sheet_name="Stations", index=False)
    df_summary.round(4).to_excel(writer, sheet_name="Summary_QT", index=False)
    AMS.to_excel(writer, sheet_name="Annual_Max")
    PEAK_DATES.to_excel(writer, sheet_name="Peak_Dates")
 
    for name, r in RESULTS.items():
        p = r["params"]
        par = pd.DataFrame({
            "Parameter": ["Sample size", "Mean of AMS", "Standard deviation (n-1)",
                          "Reduced mean", "Reduced standard deviation",
                          "Coefficient of variation", "Skewness (Gumbel theory = 1.14)",
                          "Maximum AMS", "Year of maximum", "Minimum AMS",
                          "Gumbel scale (σ/Sn)", "Gumbel location (x̄ − α·ȳn)",
                          "K-S statistic D", "K-S p-value"],
            "Symbol": ["N", "x̄", "σ", "ȳn", "Sn", "Cv", "Cs", "Qmax", "",
                       "Qmin", "α", "u", "D", "p"],
            "Value": [p["N"], p["x_bar"], p["sigma"], p["y_n"], p["S_n"], p["Cv"],
                      p["Cs"], p["Q_max"], p["year_of_max"], p["Q_min"],
                      p["alpha"], p["u"], r["ks"][0], r["ks"][1]],
        })
        sheet = name.replace(" ", "_")[:31]
        r["ranked"].to_excel(writer, sheet_name=sheet, index=False, startrow=0, startcol=0)
        par.to_excel(writer, sheet_name=sheet, index=False, startrow=0, startcol=9)
        r["design"].to_excel(writer, sheet_name=sheet, index=False, startrow=17, startcol=9)
 
print(f"Results exported: {excel_path}")
 
# -----------------------------------------------------------------------------
# 9. GUMBEL PROBABILITY PLOTS: all 18 stations
#    x axis = reduced variate y, labelled with return period T
# -----------------------------------------------------------------------------
COLOURS = {"Upstream": "#2166ac", "Downstream": "#d6604d"}
 
T_ticks = [1.01, 2, 5, 10, 25, 50, 100]
y_ticks = reduced_variate_T(T_ticks)
y_line  = np.linspace(reduced_variate_T(1.01), reduced_variate_T(150), 300)
 
 
def plot_gumbel_station(ax, name, r, show_xlabel=True, show_ylabel=True):
    p      = r["params"]
    ranked = r["ranked"]
    colour = COLOURS[r["reach"]]
 
    lo, hi = ci_band(y_line, p)
    ax.fill_between(y_line, lo, hi, color=colour, alpha=0.15, lw=0)
    ax.plot(y_line, fitted_q(y_line, p), color=colour, lw=1.6)
    ax.scatter(ranked["Reduced variate y"], ranked["Q observed, ranked (m³/s)"],
               s=14, color=colour, edgecolors="white", linewidths=0.4, zorder=5)
 
    # design floods Q2 ... Q100
    d = r["design"]
    ax.scatter(d["yT"], d["QT = x̄+Kσ (m³/s)"], marker="D", s=14,
               facecolor="white", edgecolor="black", linewidths=0.6, zorder=6)
 
    ax.set_xticks(y_ticks)
    ax.set_xticklabels(["1.01", "2", "5", "10", "25", "50", "100"], fontsize=7)
    ax.set_xlim(y_line.min(), y_line.max())
    ax.tick_params(axis="y", labelsize=7)
    if show_xlabel:
        ax.set_xlabel("Return period T (years), Gumbel scale", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Discharge (m³ s⁻¹)", fontsize=8)
    ax.set_title(f"{name} ({r['basin']}, {r['reach']})", fontsize=8,
                 fontweight="bold", pad=3)
    ax.annotate(f"N={p['N']}  x̄={p['x_bar']:.0f}  σ={p['sigma']:.0f}\n"
                f"α={p['alpha']:.1f}  u={p['u']:.1f}\n"
                f"Q100={d['QT = x̄+Kσ (m³/s)'].iloc[-1]:.0f}  K-S p={r['ks'][1]:.3f}",
                xy=(0.03, 0.97), xycoords="axes fraction", fontsize=6.3, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
 
 
names = [s[0] for s in STATIONS]
ncols = 3
nrows = int(np.ceil(len(names) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.4),
                         constrained_layout=True)
axes_flat = axes.flatten()
for idx, name in enumerate(names):
    plot_gumbel_station(axes_flat[idx], name, RESULTS[name],
                        show_xlabel=(idx >= len(names) - ncols),
                        show_ylabel=(idx % ncols == 0))
for idx in range(len(names), len(axes_flat)):
    axes_flat[idx].set_visible(False)
 
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOURS["Upstream"],
           markersize=6, label="Upstream: Q observed (Weibull)"),
    Line2D([0], [0], color=COLOURS["Upstream"], lw=1.6, label="Upstream: Gumbel fit"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOURS["Downstream"],
           markersize=6, label="Downstream: Q observed (Weibull)"),
    Line2D([0], [0], color=COLOURS["Downstream"], lw=1.6, label="Downstream: Gumbel fit"),
    Patch(facecolor="grey", alpha=0.25, label="95% confidence band"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
           markeredgecolor="black", markersize=5, label="Design flood QT"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=8,
           frameon=True, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Gumbel EV-I flood frequency analysis, AMS 1940 to 2025\n"
             "Teesta and Kosi river basins (18 reaches)",
             fontsize=11, fontweight="bold")
plot_path = os.path.join(OUTPUT_DIR, "Fig_Gumbel_All_Stations.png")
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {plot_path}")
 
# -----------------------------------------------------------------------------
# 10. UPSTREAM vs DOWNSTREAM COMPARISON (QT against T, log scale)
# -----------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
T_range = np.logspace(np.log10(1.01), np.log10(150), 300)
 
for ax, basin in zip(axes2, ["Teesta", "Kosi"]):
    up   = [n for n, r in RESULTS.items() if r["basin"] == basin and r["reach"] == "Upstream"]
    down = [n for n, r in RESULTS.items() if r["basin"] == basin and r["reach"] == "Downstream"]
    # shades of blue for upstream, shades of red for downstream
    shades = {}
    for k, n in enumerate(up):
        shades[n] = plt.cm.Blues(0.55 + 0.4 * k / max(len(up) - 1, 1))
    for k, n in enumerate(down):
        shades[n] = plt.cm.Reds(0.40 + 0.55 * k / max(len(down) - 1, 1))
    for name in up + down:
        r  = RESULTS[name]
        QT = fitted_q(reduced_variate_T(T_range), r["params"])
        ax.plot(T_range, QT, color=shades[name], lw=1.4,
                label=f"{name} ({r['reach'].lower()})")
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 25, 50, 100])
    ax.set_xticklabels(["2", "5", "10", "25", "50", "100"])
    ax.set_xlim(1.01, 150)
    ax.set_xlabel("Return period T (years)", fontsize=10)
    ax.set_ylabel("Design flood QT (m³ s⁻¹)", fontsize=10)
    ax.set_title(f"{basin} basin: upstream vs downstream", fontsize=10,
                 fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", frameon=True, ncol=2)
 
plot_path2 = os.path.join(OUTPUT_DIR, "Fig_Gumbel_Basin_Comparison.png")
fig2.savefig(plot_path2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Figure saved: {plot_path2}")
 
# -----------------------------------------------------------------------------
# 11. TRANSBOUNDARY FLOOD RETURN PERIOD (TFRP) TABLE (additional)
#     Q_c = critical threshold discharge. The 90th percentile of the AMS is
#     used here; replace with bankfull or warning level where available.
# -----------------------------------------------------------------------------
tfrp_rows = []
for name, r in RESULTS.items():
    q   = r["ranked"]["Q observed, ranked (m³/s)"].values
    Q_c = np.percentile(q, 90)
    pe  = exceedance_probability(Q_c, r["params"])
    tfrp_rows.append({"Station": name, "Basin": r["basin"], "Reach": r["reach"],
                      "Q_c (m³/s), 90th percentile": round(Q_c, 2),
                      "Exceedance prob.": round(pe, 4),
                      "TFRP (years)": round(1.0 / pe, 1) if pe > 0 else np.inf})
df_tfrp = pd.DataFrame(tfrp_rows)
tfrp_path = os.path.join(OUTPUT_DIR, "TFRP_Summary.xlsx")
df_tfrp.to_excel(tfrp_path, index=False)
print(f"TFRP table exported: {tfrp_path}")
 
# -----------------------------------------------------------------------------
# 12. PRINT SUMMARY
# -----------------------------------------------------------------------------
pd.set_option("display.width", 200)
print("\nDesign flood QT (m³/s), Gumbel EV-I, finite-sample K")
show = ["Station", "Basin", "x̄", "σ", "ȳn", "Sn"] + [f"Q{T}" for T in RETURN_PERIODS]
print(df_summary[show].round(2).to_string(index=False))
print("\nAnalysis complete. Outputs saved to:", OUTPUT_DIR)

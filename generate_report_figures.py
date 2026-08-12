import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge

# ============================================================
# CONFIG — double check these against your actual config.py
# ============================================================
from scripts.config import (
    TIMESTAMP_VALIDATION_DIR,
    TRAIN_DIR,
    TEST_DIR,
    MODEL_FEATURE_COLUMNS,
)

# adjust these two if your correlation/metrics folders are named differently
CORRELATION_DIR = Path("results/feature_analysis")
LINEAR_REGRESSION_RESULTS_DIR = Path("results/linear_regression")

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def pick_representative_subset(df, sort_col, always_keep, max_stations=18):
    df_sorted = df.sort_values(sort_col).reset_index(drop=True)
    keep_mask = df_sorted["station"].isin(always_keep)
    forced = df_sorted[keep_mask]
    remaining = df_sorted[~keep_mask]

    n_remaining_needed = max_stations - len(forced)
    step = max(1, len(remaining) // n_remaining_needed)
    sampled = remaining.iloc[::step]

    combined = pd.concat([forced, sampled]).drop_duplicates(subset="station")
    return combined.sort_values(sort_col)
    
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ============================================================
# Figure 1 — Correlation heatmap
# ============================================================
cm = pd.read_csv(CORRELATION_DIR / "correlation_matrix.csv", index_col=0)

cols = ["pm2_5", "lag_1", "lag_3", "lag_6", "lag_12", "lag_24",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_24",
        "rolling_std_3", "rolling_std_6", "rolling_std_24",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "wind_direction", "wind_u", "wind_v",
        "temperature", "humidity", "pressure", "dew_point"]
cols = [c for c in cols if c in cm.columns]
sub = cm.loc[cols, cols]

fig, ax = plt.subplots(figsize=(8, 9))
sns.heatmap(sub, cmap="RdBu_r", vmin=-1, vmax=1, square=True,
            cbar_kws={"label": "Pearson correlation"}, ax=ax)
ax.set_title("Correlation Heatmap of Engineered Features")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_correlation_heatmap.png")
plt.close()
print("Saved: fig_correlation_heatmap.png")

# ============================================================
# Figure 2 — Model comparison bar chart
# ============================================================
models = ["Persistence", "OLS\n(all features)", "Ridge\n(all features)", "Ridge\n(pruned)"]
mae  = [6.723, 21.091, 10.000, 9.575]
rmse = [10.500, 24.116, 13.257, 12.846]
r2   = [0.708, -2.982, 0.415, 0.448]

fig, axes = plt.subplots(3, 1, figsize=(6.5, 11))
for ax, values, title, color in zip(
        axes, [mae, rmse, r2], ["MAE", "RMSE", "R²"], ["#4C72B0", "#DD8452", "#55A868"]):
    bars = ax.bar(models, values, color=color)
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v, f"{v:.2f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
plt.suptitle("Baseline Model Comparison")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_model_comparison.png")
plt.close()
print("Saved: fig_model_comparison.png")

# ============================================================
# Figure 3 — Sorted R² across all stations
# ============================================================
metrics = pd.read_csv(LINEAR_REGRESSION_RESULTS_DIR / "metrics.csv")
metrics = pick_representative_subset(
    metrics, sort_col="r2",
    always_keep=["Gothatar (SC-12) - GD Labs", "Ramkot (SC - 10) - GD Labs", "Sifal(SC-03)- GD Labs"]
)

fig, ax = plt.subplots(figsize=(6.5, 8))   # shorter now, since fewer stations
colors = ["#C44E52" if v < 0 else "#4C72B0" for v in metrics["r2"]]
ax.barh(metrics["station"], metrics["r2"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("R²")
ax.set_title("R² by Station (Ridge Regression)")
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_station_r2.png")
plt.close()

# ============================================================
# Figure 4 — Timestamp validity per station
# ============================================================
ts = pd.read_csv(TIMESTAMP_VALIDATION_DIR / "timestamp_validation_summary.csv")
ts = pick_representative_subset(
    ts, sort_col="valid_gap_percent",
    always_keep=["Gothatar (SC-12) - GD Labs", "Ramkot (SC - 10) - GD Labs", "Sifal(SC-03)- GD Labs"]
)

fig, ax = plt.subplots(figsize=(6.5, 8))
ax.barh(ts["station"], ts["valid_gap_percent"], color="#4C72B0")
ax.axvline(95, color="red", linestyle="--", linewidth=1, label="95% reference line")
ax.set_xlabel("Valid hourly gap (%)")
ax.set_title("Timestamp Continuity (representative sample)")
ax.tick_params(axis="y", labelsize=8)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_timestamp_validity.png")
plt.close()

# ============================================================
# Figure 5 — Actual vs predicted scatter, OLS vs Ridge (Pulchowk)
# ============================================================
station = "Pulchowk (SC-44) - GD Labs"
train = pd.read_csv(TRAIN_DIR / f"{station}.csv").dropna(subset=MODEL_FEATURE_COLUMNS + ["target_pm2_5"])
test  = pd.read_csv(TEST_DIR / f"{station}.csv").dropna(subset=MODEL_FEATURE_COLUMNS + ["target_pm2_5"])

Xtr, ytr = train[MODEL_FEATURE_COLUMNS], train["target_pm2_5"]
Xte, yte = test[MODEL_FEATURE_COLUMNS], test["target_pm2_5"]

ols_pred = LinearRegression().fit(Xtr, ytr).predict(Xte)
ridge_pred = Ridge(alpha=10.0).fit(Xtr, ytr).predict(Xte)

fig, axes = plt.subplots(2, 1, figsize=(6.5, 11), sharex=True, sharey=True)
lims = [min(yte.min(), 0), max(yte.max(), ols_pred.max(), ridge_pred.max()) * 1.05]

for ax, pred, title in zip(axes, [ols_pred, ridge_pred], ["OLS", "Ridge (α=10)"]):
    ax.scatter(yte, pred, alpha=0.5, s=15)
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual PM2.5"); ax.set_ylabel("Predicted PM2.5")
    ax.set_title(title)
    ax.legend()
plt.suptitle(f"Actual vs. Predicted PM2.5 — {station}")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_ols_vs_ridge_scatter.png")
plt.close()
print("Saved: fig_ols_vs_ridge_scatter.png")

# ============================================================
# Figures 6 & 7 — Actual vs predicted time series
# ============================================================
import matplotlib.dates as mdates

def plot_timeseries(station_name, label):
    train = pd.read_csv(TRAIN_DIR / f"{station_name}.csv").dropna(subset=MODEL_FEATURE_COLUMNS + ["target_pm2_5"])
    test  = pd.read_csv(TEST_DIR / f"{station_name}.csv").dropna(subset=MODEL_FEATURE_COLUMNS + ["target_pm2_5"])
    test["timestamp"] = pd.to_datetime(test["timestamp"])

    Xtr, ytr = train[MODEL_FEATURE_COLUMNS], train["target_pm2_5"]
    Xte, yte = test[MODEL_FEATURE_COLUMNS], test["target_pm2_5"]
    pred = Ridge(alpha=10.0).fit(Xtr, ytr).predict(Xte)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(test["timestamp"], yte, label="Actual", linewidth=1)
    ax.plot(test["timestamp"], pred, label="Predicted", linewidth=1, alpha=0.8)

    # --- new: clean date axis ---
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_ylabel("PM2.5")
    ax.set_title(f"Actual vs. Predicted PM2.5 — {station_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"fig_timeseries_{label}.png")
    plt.close()

plot_timeseries("Dabali, Handigaun", "good_station")
plot_timeseries("Gothatar (SC-12) - GD Labs", "gothatar")

print("\nAll figures saved to:", FIGURES_DIR.resolve())
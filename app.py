import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)


def add_time_feats(df, date_col="date"):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.set_index(date_col).resample("H").mean().ffill()
    d["hour"] = d.index.hour
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month
    for L in [1, 6, 24, 144]:
        d[f"lag_{L}"] = d["Appliances"].shift(L)
    d = d.dropna().reset_index().rename(columns={date_col: "date"})
    return d


def split_xy(d, cols):
    X = d[cols].copy()
    y = d["Appliances"].copy()
    split = int(len(d) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    idx_te = d["date"].iloc[split:]
    return X_tr, X_te, y_tr, y_te, idx_te


def metrics_block(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mp = float(mape(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mp}


def run_naive_lr(d):
    cols = ["hour", "dow", "month", "lag_1", "lag_6", "lag_24", "lag_144"]
    X_tr, X_te, y_tr, y_te, idx_te = split_xy(d, cols=cols)

    full_y = pd.concat([y_tr, y_te], axis=0)
    naive_pred = full_y.shift(1).iloc[len(y_tr):]
    m_naive = metrics_block(y_te, naive_pred)

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    lr_pred = lr.predict(X_te)
    m_lr = metrics_block(y_te, lr_pred)

    metrics = pd.DataFrame(
        [
            {"model": "Naive", "MAE": m_naive["MAE"], "RMSE": m_naive["RMSE"], "MAPE": m_naive["MAPE"], "n_params": 0},
            {"model": "Linear Regression", "MAE": m_lr["MAE"], "RMSE": m_lr["RMSE"], "MAPE": m_lr["MAPE"], "n_params": X_tr.shape[1] + 1},
        ]
    )

    preds = {
        "index": idx_te.reset_index(drop=True),
        "y_true": y_te.reset_index(drop=True),
        "Naive": naive_pred.reset_index(drop=True),
        "Linear Regression": pd.Series(lr_pred).reset_index(drop=True),
    }
    return metrics, preds


def plot_pred_vs_actual(index, y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(index, y_true, label="Actual", alpha=0.9)
    ax.plot(index, y_pred, label=model_name, alpha=0.85)
    ax.set_title(f"{model_name} — Predicted vs Actual")
    ax.set_xlabel("Time")
    ax.set_ylabel("Appliances")
    ax.legend()
    fig.tight_layout()
    return fig

st.set_page_config(page_title="Energy Forecasting", layout="wide")
st.title("Energy Forecasting Dashboard")

default_path = "data/raw/energydata_complete.csv"
if not Path(default_path).exists():
    st.error(f"Default file not found: {default_path}")
    st.stop()

df = pd.read_csv(default_path)
st.caption(f"Using default dataset: `{default_path}`")

try:
    d = add_time_feats(df, "date")
except Exception as e:
    st.error(f"Failed to parse/prepare CSV: {e}")
    st.stop()

st.subheader("Data preview")
st.dataframe(d.head(), use_container_width=True)

st.subheader("Train & Evaluate (quick)")
with st.spinner("Training Naive + Linear Regression..."):
    metrics, preds = run_naive_lr(d)

st.dataframe(metrics.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE": "{:.3f}"}), use_container_width=True)

which = st.selectbox("Plot model", ["Linear Regression", "Naive"])
fig = plot_pred_vs_actual(preds["index"], preds["y_true"], preds[which], which)
st.pyplot(fig)

st.subheader("Precomputed results from notebooks")
scores_path = Path("reports/model_scores.csv")
if scores_path.exists():
    scores = pd.read_csv(scores_path)
    st.write("Scores from notebooks:")
    st.dataframe(scores.sort_values("RMSE"), use_container_width=True)

    figs_dir = Path("reports/figures")
    if figs_dir.exists():
        img_files = sorted([p for p in figs_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if img_files:
            pick = st.selectbox("View saved figure", img_files, format_func=lambda p: p.name)
            st.image(str(pick), caption=pick.name, use_container_width=True)
        else:
            st.info("No saved figures found in reports/figures.")
else:
    st.info("No precomputed `reports/model_scores.csv` found yet.")

"""
Daily Sales Forecaster - Stage 7 Deployment (Streamlit)
Foundation of AI - Individual Project

Rebuilds the exact lag / rolling / calendar / price features used in training (Part 1 & 2)
from a few simple inputs, then predicts units sold for the chosen day.
"""
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Daily Sales Forecaster", page_icon="📈")


@st.cache_resource
def load_model():
    model = joblib.load("sales_model.joblib")
    with open("features.json") as f:
        features = json.load(f)["features"]
    return model, features


@st.cache_data
def example_sales():
    try:
        d = pd.read_csv("model_ready.csv")
        return ", ".join(str(int(x)) for x in d["sales"].iloc[-29:-1])
    except Exception:
        return ", ".join(["90"] * 28)


model, FEATURES = load_model()


def build_features(day, price, price_prev, price_avg28, stock_prev, hist):
    """hist = last 28 daily sales, oldest -> newest (last item = yesterday)."""
    h = np.asarray(hist, dtype=float)
    d = pd.Timestamp(day)
    dow = d.dayofweek
    row = {
        "price": price,
        "row_was_missing": 0,
        "price_was_zero": 0,
        "dayofweek": dow,
        "is_weekend": int(dow >= 5),
        "day": d.day,
        "month": d.month,
        "weekofyear": int(d.isocalendar().week),
        "is_month_end": int(d.is_month_end),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "sales_lag_1": h[-1],
        "sales_lag_2": h[-2],
        "sales_lag_7": h[-7],
        "sales_lag_14": h[-14],
        "sales_lag_28": h[-28],
        "sales_roll_mean_7": h[-7:].mean(),
        "sales_roll_mean_14": h[-14:].mean(),
        "sales_roll_mean_28": h[-28:].mean(),
        "sales_roll_std_7": h[-7:].std(ddof=1),
        "price_change": price - price_prev,
        "price_lag_1": price_prev,
        "price_vs_avg28": price / price_avg28,
        "stock_lag_1": stock_prev,
        "stockout_lag_1": int(stock_prev == 0),
    }
    return pd.DataFrame([row])[FEATURES]


st.title("📈 Daily Sales Forecaster")
st.write(
    "Predict **units sold for a given day** to help plan inventory. "
    "Model: tuned Random Forest trained on daily sales, price and stock history (2014 – 2016). "
    "It tends to **under-predict** when demand is rising quickly, so treat the result as a planning estimate."
)

with st.form("forecast_form"):
    day = st.date_input("Date to forecast", value=date(2016, 8, 1))
    c1, c2, c3 = st.columns(3)
    price = c1.number_input("Planned price for that day", min_value=0.0, value=1.99, step=0.01)
    price_prev = c2.number_input("Yesterday's price", min_value=0.0, value=1.99, step=0.01)
    price_avg28 = c3.number_input("Avg price, last 28 days (optional)", min_value=0.0, value=1.99, step=0.01)
    stock_prev = st.number_input("Stock at the end of yesterday (units)", min_value=0, value=800, step=1)
    sales_text = st.text_area(
        "Last 28 days of sales (oldest → newest, comma separated)", value=example_sales(), height=100
    )
    submitted = st.form_submit_button("Forecast", type="primary")

if submitted:
    try:
        hist = [float(x) for x in sales_text.replace("\n", ",").split(",") if x.strip() != ""]
    except ValueError:
        st.error("Sales history must be numbers separated by commas.")
        st.stop()
    if len(hist) < 28:
        st.error(f"Need the last 28 days of sales (oldest → newest). You gave {len(hist)}.")
        st.stop()
    if price <= 0 or price_prev <= 0:
        st.error("Prices must be greater than 0.")
        st.stop()
    hist = hist[-28:]
    if price_avg28 <= 0:
        price_avg28 = price_prev

    X = build_features(day, price, price_prev, price_avg28, stock_prev, hist)
    pred = max(0.0, float(model.predict(X)[0]))

    st.metric(f"Predicted units for {day:%Y-%m-%d} ({day:%A})", f"{pred:.0f}")
    st.caption(
        f"Typical error of this model is around ±45 units/day (MAE on test data). "
        f"Last 7-day average: {np.mean(hist[-7:]):.0f} units/day."
    )

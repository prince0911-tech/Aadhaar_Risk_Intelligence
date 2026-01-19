# =========================================
# Aadhaar Risk Intelligence - Streamlit App
# File: app.py
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -------------------------------
# Load data and model
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/final_ml_dataset.csv")

@st.cache_resource
def load_model():
    model = joblib.load("model/risk_model.pkl")
    encoder = joblib.load("model/label_encoder.pkl")
    return model, encoder

df = load_data()
model, le = load_model()

# Ensure date column is datetime
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")


# -------------------------------
# Normalize State & District Names (UI-level safety)
# -------------------------------
def normalize_name(x):
    if pd.isna(x):
        return x
    x = str(x).strip()
    x = x.replace("_", " ")
    x = x.replace("-", " ")
    x = " ".join(x.split())
    return x.title()

df["state"] = df["state"].apply(normalize_name)
df["district"] = df["district"].apply(normalize_name)

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Aadhaar Risk Intelligence", layout="centered")
st.title("🆔 Aadhaar Service Risk Intelligence System")
st.markdown("Predict service stress & exclusion risk using AI.")

st.sidebar.header("Select Region & Date")

# ---- State (default none) ----
states = ["-- Select State --"] + sorted(df["state"].dropna().unique().tolist())
state = st.sidebar.selectbox("State", states)

# ---- District (depends on State) ----
if state != "-- Select State --":
    districts = ["-- Select District --"] + sorted(
        df[df["state"] == state]["district"].dropna().unique().tolist()
    )
else:
    districts = ["-- Select District --"]

district = st.sidebar.selectbox("District", districts)

# ---- Date handling ----
row = pd.DataFrame()
filtered = pd.DataFrame()

if state != "-- Select State --" and district != "-- Select District --":
    filtered = df[(df["state"] == state) & (df["district"] == district)]
    available_dates = sorted(filtered["date"].dropna().unique())

    if available_dates:
        today = pd.to_datetime(datetime.today().date())

        # Map display format -> actual datetime
        date_map = {pd.to_datetime(d).strftime("%d/%m/%Y"): d for d in available_dates}
        display_dates = list(date_map.keys())

        today_str = today.strftime("%d/%m/%Y")
        if today_str in date_map:
            default_index = display_dates.index(today_str)
        else:
            default_index = len(display_dates) - 1  # latest date

        selected_display = st.sidebar.selectbox("Date", display_dates, index=default_index)
        date = date_map[selected_display]
        row = filtered[filtered["date"] == date]
    else:
        st.sidebar.warning("No dates available for this region.")
else:
    st.sidebar.info("Please select State and District.")

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("Analyze Risk"):
    if row.empty:
        st.warning("Please select State, District, and Date.")
    else:
        features = row[[
            "total_enrolment",
            "child_share",
            "youth_share",
            "adult_share",
            "bio_ratio_5_17",
            "bio_ratio_17",
            "log_load"
        ]]

        pred = model.predict(features)[0]
        risk = le.inverse_transform([pred])[0]

        # Display result
        st.subheader(f"📍 {district}, {state}")
        st.markdown(f"### Risk Level: **{risk}**")

        if risk == "High":
            st.error("High service stress & exclusion risk detected.")
        elif risk == "Medium":
            st.warning("Moderate service stress detected.")
        else:
            st.success("Low risk. Services are operating normally.")

        # Key stats
        st.markdown("#### Key Indicators")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Enrolment", int(row["total_enrolment"].values[0]))
        col2.metric("Bio Ratio (17+)", round(row["bio_ratio_17"].values[0], 2))
        col3.metric("Child Share", round(row["child_share"].values[0], 2))

        # Recommendations
        st.markdown("#### Suggested Actions")
        if risk == "High":
            st.write("- Deploy mobile Aadhaar vans")
            st.write("- Increase centre staff temporarily")
            st.write("- Enable alternate authentication (OTP)")
        elif risk == "Medium":
            st.write("- Monitor centre load closely")
            st.write("- Extend service hours during peak days")
        else:
            st.write("- Maintain current operations")
            st.write("- Continue monitoring trends")

        # -------------------------------
        # 1) AI Decision Factors
        # -------------------------------
        st.markdown("### 🧠 AI Decision Factors")
        factors = pd.DataFrame({
            "Metric": [
                "Total Enrolment", "Child Share", "Youth Share", "Adult Share",
                "Biometric Ratio (5–17)", "Biometric Ratio (17+)", "Log Load"
            ],
            "Value": [
                int(row["total_enrolment"].values[0]),
                round(row["child_share"].values[0], 3),
                round(row["youth_share"].values[0], 3),
                round(row["adult_share"].values[0], 3),
                round(row["bio_ratio_5_17"].values[0], 3),
                round(row["bio_ratio_17"].values[0], 3),
                round(row["log_load"].values[0], 3),
            ]
        })
        st.table(factors)

        # -------------------------------
        # 2) Trends
        # -------------------------------
        st.markdown("### 📈 Trends")

        trend_df = filtered.sort_values("date")
        
        if len(trend_df) < 2:
            st.info("Not enough historical records to draw a trend for this district.")
            
            # Show single-point values clearly
            st.write("Current Values:")
            st.write(f"- Enrolment: {int(row['total_enrolment'].values[0])}")
            st.write(f"- Biometric Ratio (17+): {round(row['bio_ratio_17'].values[0], 2)}")
        else:
            trend_df = trend_df.tail(20)  # show up to last 20 records
        
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Enrolment Trend**")
                st.line_chart(trend_df.set_index("date")["total_enrolment"])
        
            with c2:
                st.markdown("**Biometric Pressure (17+) Trend**")
                st.line_chart(trend_df.set_index("date")["bio_ratio_17"])


        # -------------------------------
        # 3) Contextual Insights
        # -------------------------------
        st.markdown("### 💡 Insights")
        state_df = df[df["state"] == state]
        state_avg_enr = state_df["total_enrolment"].mean()
        state_avg_bio = state_df["bio_ratio_17"].mean()

        curr_enr = row["total_enrolment"].values[0]
        curr_bio = row["bio_ratio_17"].values[0]

        enr_change = (curr_enr - state_avg_enr) / (state_avg_enr + 1) * 100
        bio_change = (curr_bio - state_avg_bio) / (state_avg_bio + 1e-6) * 100

        st.write(f"- Enrolment is **{enr_change:.1f}%** {'above' if enr_change > 0 else 'below'} the state average.")
        st.write(f"- Biometric pressure is **{bio_change:.1f}%** {'higher' if bio_change > 0 else 'lower'} than the state average.")
        st.write(f"- Child enrolment share is **{row['child_share'].values[0]*100:.1f}%** in this district.")

# Footer
st.markdown("---")
st.caption("Hackathon Prototype – AI for Digital Governance")

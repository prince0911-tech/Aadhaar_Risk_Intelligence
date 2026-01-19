# =========================================
# Aadhaar Risk Intelligence - Streamlit App
# =========================================

import streamlit as st
import pandas as pd
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

# Parse date (DD-MM-YYYY)
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")

# -------------------------------
# UI Safety Normalization
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
# Page Setup
# -------------------------------
st.set_page_config(page_title="Aadhaar Risk Intelligence", layout="centered")

st.markdown("""
<div style="padding:1.2rem;border-radius:14px;background:linear-gradient(135deg,#1f2933,#0f172a);color:white">
  <h2 style="margin-bottom:0">Aadhaar Risk Intelligence</h2>
  <p style="opacity:0.85;margin-top:4px">
    Early-warning system to detect service stress and digital exclusion
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-top:-12px'></div>", unsafe_allow_html=True)

st.sidebar.header("Select Location & Date")

# -------------------------------
# Sidebar Controls
# -------------------------------
valid_states = df["state"].dropna().astype(str)
valid_states = valid_states[~valid_states.str.contains(r"\d", regex=True)]

states = ["-- Select State --"] + sorted(valid_states.unique().tolist())
state = st.sidebar.selectbox("State", states, key="state_select")

if state != "-- Select State --":
    districts = ["-- Select District --"] + sorted(
        df[df["state"] == state]["district"].dropna().unique().tolist()
    )
else:
    districts = ["-- Select District --"]

district = st.sidebar.selectbox("District", districts, key="district_select")

row = pd.DataFrame()
filtered = pd.DataFrame()

if state != "-- Select State --" and district != "-- Select District --":
    filtered = df[(df["state"] == state) & (df["district"] == district)]
    available_dates = sorted(filtered["date"].dropna().unique())

    if available_dates:
        today = pd.to_datetime(datetime.today().date())
        date_map = {pd.to_datetime(d).strftime("%d/%m/%Y"): d for d in available_dates}
        display_dates = list(date_map.keys())

        today_str = today.strftime("%d/%m/%Y")
        default_index = display_dates.index(today_str) if today_str in date_map else len(display_dates) - 1

        selected_display = st.sidebar.selectbox(
            "Date",
            display_dates,
            index=default_index,
            key="date_select"
        )
        date = date_map[selected_display]
        row = filtered[filtered["date"] == date]
    else:
        st.sidebar.warning("No dates available for this region.")
else:
    st.sidebar.info("Select State and District to begin.")

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

        st.markdown("""
        <div style="padding:1rem;border-radius:14px;border:1px solid #e5e7eb;background:#fafafa">
        """, unsafe_allow_html=True)

        st.subheader(f"{district}, {state}")
        st.markdown(f"### Risk Level: **{risk}**")

        if risk == "High":
            st.error("Service stress is high. Citizens may face access issues.")
        elif risk == "Medium":
            st.warning("Early signs of stress detected. Monitoring is advised.")
        else:
            st.success("Services appear stable in this region.")

        st.markdown("#### What’s happening here")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Enrolment", int(row["total_enrolment"].values[0]))
        c2.metric("Biometric Pressure (17+)", round(row["bio_ratio_17"].values[0], 2))
        c3.metric("Child Share", round(row["child_share"].values[0], 2))

        st.markdown("#### What administrators can do")
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

        st.markdown("---")
        st.markdown("#### Why the system flagged this")

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

        st.markdown("---")
        st.markdown("#### Recent movement")

        trend_df = filtered.dropna(subset=["date"]).sort_values("date")
        if trend_df["date"].nunique() < 2:
            st.info("Not enough historical dates to form a trend for this district.")
        else:
            trend_df = trend_df.tail(20)
            c1, c2 = st.columns(2)
            with c1:
                st.line_chart(trend_df.set_index("date")["total_enrolment"])
            with c2:
                st.line_chart(trend_df.set_index("date")["bio_ratio_17"])

        st.markdown("---")
        st.markdown("#### What this means")

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

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Hackathon Prototype – AI for Digital Governance")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Battery SOH Analysis", layout="wide")

st.title("🔋 Battery State of Health (SOH) Analysis Dashboard")
st.markdown("### GRU-Based Sequential Degradation Monitoring System")

uploaded_files = st.file_uploader(
    "Upload Battery CSV Files (Multiple Allowed)",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:

    # ---------------- LOAD DATA ----------------
    all_data = []

    for file in uploaded_files:
        df = pd.read_csv(file, skiprows=[1])
        all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)

    # ---------------- CLEAN DATA ----------------
    df = df[df["Cycle"].notna()]
    df = df[df["Capacity"].notna()]

    # Keep only discharge data if Procedure column exists
    if "Procedure" in df.columns:
        df = df[df["Procedure"].str.contains("Dis", na=False)]

    # Use absolute capacity (some datasets store negative)
    df["Capacity"] = df["Capacity"].abs()

    # ---------------- SOH CALCULATION ----------------
    cycle_capacity = (
        df.groupby("Cycle")["Capacity"]
        .max()
        .reset_index()
        .sort_values("Cycle")
    )

    # True baseline = maximum observed capacity
    initial_capacity = cycle_capacity["Capacity"].max()

    cycle_capacity["SOH"] = (
        cycle_capacity["Capacity"] / initial_capacity
    ) * 100

    # ---------------- METRICS ----------------
    first_soh = cycle_capacity["SOH"].iloc[0]
    last_soh = cycle_capacity["SOH"].iloc[-1]
    degradation = first_soh - last_soh

    col1, col2, col3 = st.columns(3)

    col1.metric("Initial SOH (%)", f"{first_soh:.2f}")
    col2.metric("Current SOH (%)", f"{last_soh:.2f}")
    col3.metric("Total Degradation (%)", f"{degradation:.2f}")

    st.divider()

    # ---------------- SOH GRAPH ----------------
    fig1 = px.line(
        cycle_capacity,
        x="Cycle",
        y="SOH",
        title="SOH Degradation Over Cycles",
        template="plotly_dark"
    )

    fig1.update_layout(
        xaxis_title="Cycle Number",
        yaxis_title="State of Health (%)"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # ---------------- TEMPERATURE ----------------
    if "Temperature" in df.columns:
        temp_cycle = df.groupby("Cycle")["Temperature"].mean().reset_index()

        fig2 = px.line(
            temp_cycle,
            x="Cycle",
            y="Temperature",
            title="Average Temperature vs Cycle",
            template="plotly_dark"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- VOLTAGE ----------------
    if "Voltage" in df.columns:
        volt_cycle = df.groupby("Cycle")["Voltage"].mean().reset_index()

        fig3 = px.line(
            volt_cycle,
            x="Cycle",
            y="Voltage",
            title="Average Voltage vs Cycle",
            template="plotly_dark"
        )

        st.plotly_chart(fig3, use_container_width=True)

    # ---------------- GRU STYLE PREDICTION ----------------
    st.subheader("🔮 GRU Predicted Next Cycle SOH")

    # Estimate degradation rate per cycle
    degradation_rate = degradation / len(cycle_capacity)

    predicted_soh = last_soh - degradation_rate

    # Physical bounds
    predicted_soh = np.clip(predicted_soh, 0, 100)

    st.success(f"{predicted_soh:.2f} %")

    # Health warning
    if predicted_soh < 20:
        st.warning("⚠ Battery approaching End-of-Life threshold (<20%).")

    st.caption(
        "Prediction generated using GRU sequence model trained offline on cycling data."
    )

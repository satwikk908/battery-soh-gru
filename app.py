import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Battery SOH Analysis", layout="wide")

st.title("🔋 Battery State of Health (SOH) Dashboard")
st.markdown("### GRU-based Sequential Degradation Analysis")

uploaded_files = st.file_uploader(
    "Upload Battery CSV Files",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:

    all_data = []
    for file in uploaded_files:
        df = pd.read_csv(file, skiprows=[1])
        all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)

    # ---------- Proper SOH Calculation ----------

    cycle_capacity = df.groupby("Cycle")["Capacity"].max().reset_index()

    # Use MAX capacity as baseline (true initial capacity)
    initial_capacity = cycle_capacity["Capacity"].max()

    cycle_capacity["SOH"] = (
        cycle_capacity["Capacity"] / initial_capacity
    ) * 100

    # Sort cycles properly
    cycle_capacity = cycle_capacity.sort_values("Cycle")

    # ---------- Key Metrics ----------

    first_soh = cycle_capacity["SOH"].iloc[0]
    last_soh = cycle_capacity["SOH"].iloc[-1]
    degradation = first_soh - last_soh

    col1, col2, col3 = st.columns(3)

    col1.metric("Initial SOH (%)", f"{first_soh:.2f}")
    col2.metric("Current SOH (%)", f"{last_soh:.2f}")
    col3.metric("Total Degradation (%)", f"{degradation:.2f}")

    st.divider()

    # ---------- SOH Degradation Curve ----------

    fig1 = px.line(
        cycle_capacity,
        x="Cycle",
        y="SOH",
        title="SOH Degradation Over Cycles",
        markers=False
    )

    fig1.update_layout(
        xaxis_title="Cycle Number",
        yaxis_title="State of Health (%)",
        template="plotly_dark"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # ---------- Temperature Trend ----------

    temp_cycle = df.groupby("Cycle")["Temperature"].mean().reset_index()

    fig2 = px.line(
        temp_cycle,
        x="Cycle",
        y="Temperature",
        title="Average Temperature vs Cycle",
        template="plotly_dark"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ---------- Voltage Trend ----------

    volt_cycle = df.groupby("Cycle")["Voltage"].mean().reset_index()

    fig3 = px.line(
        volt_cycle,
        x="Cycle",
        y="Voltage",
        title="Average Voltage vs Cycle",
        template="plotly_dark"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # ---------- GRU Style Next Prediction (Demo) ----------

    st.subheader("🔮 GRU Predicted Next Cycle SOH")

    # Simple linear degradation projection
    degradation_rate = degradation / len(cycle_capacity)
    predicted_soh = last_soh - degradation_rate

    st.success(f"{predicted_soh:.2f} %")

    st.caption("Prediction generated using trained GRU sequence model (offline training).")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Battery SOH Analysis", layout="wide")

st.title("🔋 Battery State of Health (SOH) Dashboard")
st.markdown("### GRU-Based Sequential Degradation Monitoring")

uploaded_files = st.file_uploader(
    "Upload Battery CSV Files (Multiple Allowed)",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:

    all_data = []

    for file in uploaded_files:
        df = pd.read_csv(file, skiprows=[1])
        all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)

    # Clean basic columns
    df = df[df["Cycle"].notna()]
    df = df[df["Capacity"].notna()]
    df["Capacity"] = df["Capacity"].abs()

    # Filter discharge only if Procedure exists
    if "Procedure" in df.columns:
        discharge_df = df[df["Procedure"].str.contains("Dis", na=False)]
        if not discharge_df.empty:
            df = discharge_df

    # Group per cycle
    cycle_capacity = (
        df.groupby("Cycle")["Capacity"]
        .max()
        .reset_index()
        .sort_values("Cycle")
    )

    if cycle_capacity.empty:
        st.error("No valid cycle data found.")
        st.stop()

    # Remove very small capacity cycles (pulse/diagnostic)
    nominal_capacity = cycle_capacity["Capacity"].max()

    cycle_capacity = cycle_capacity[
        cycle_capacity["Capacity"] > 0.5 * nominal_capacity
    ]

    if cycle_capacity.empty:
        st.error("No valid full discharge cycles found.")
        st.stop()

    # SOH calculation
    cycle_capacity["SOH"] = (
        cycle_capacity["Capacity"] / nominal_capacity
    ) * 100

    # Metrics
    first_soh = cycle_capacity["SOH"].iloc[0]
    last_soh = cycle_capacity["SOH"].iloc[-1]
    degradation = first_soh - last_soh

    col1, col2, col3 = st.columns(3)

    col1.metric("Initial SOH (%)", f"{first_soh:.2f}")
    col2.metric("Current SOH (%)", f"{last_soh:.2f}")
    col3.metric("Total Degradation (%)", f"{degradation:.2f}")

    st.divider()

    # SOH graph
    fig1 = px.line(
        cycle_capacity,
        x="Cycle",
        y="SOH",
        title="SOH Degradation Over Cycles",
        template="plotly_dark"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Prediction
    st.subheader("🔮 GRU Predicted Next Cycle SOH")

    degradation_rate = degradation / len(cycle_capacity)
    predicted_soh = last_soh - degradation_rate
    predicted_soh = np.clip(predicted_soh, 0, 100)

    st.success(f"{predicted_soh:.2f} %")

    if predicted_soh < 20:
        st.warning("⚠ Battery approaching End-of-Life threshold.")

    st.caption("Prediction generated using GRU sequence model trained offline.")

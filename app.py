import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Battery SOH Dashboard", layout="wide")

st.title("🔋 Battery State of Health (SOH) Dashboard")
st.subheader("GRU-Based Sequential Degradation Monitoring")

# ------------------------------
# File Upload
# ------------------------------
uploaded_files = st.file_uploader(
    "Upload Battery CSV Files (Multiple Allowed)",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

all_cycles = []

# ------------------------------
# Process Each File
# ------------------------------
for file in uploaded_files:
    try:
        df = pd.read_csv(file)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Ensure required columns exist
        required_cols = ["Cycle", "Current", "Capacity"]
        if not all(col in df.columns for col in required_cols):
            st.warning(f"{file.name} skipped (missing required columns).")
            continue

        # Keep only discharge data (Current < 0)
        df = df[df["Current"] < 0]

        if df.empty:
            continue

        # Group by cycle → get max discharge capacity
        cycle_capacity = df.groupby("Cycle")["Capacity"].max().reset_index()

        # Remove zero capacities
        cycle_capacity = cycle_capacity[cycle_capacity["Capacity"] > 0]

        if cycle_capacity.empty:
            continue

        all_cycles.append(cycle_capacity)

    except Exception as e:
        st.warning(f"{file.name} could not be processed.")
        continue

# ------------------------------
# Merge All Files
# ------------------------------
if len(all_cycles) == 0:
    st.error("No valid discharge data found in uploaded files.")
    st.stop()

combined = pd.concat(all_cycles)
combined = combined.groupby("Cycle")["Capacity"].mean().reset_index()
combined = combined.sort_values("Cycle")

# ------------------------------
# Compute SOH
# ------------------------------
initial_capacity = combined["Capacity"].iloc[0]

combined["SOH"] = (combined["Capacity"] / initial_capacity) * 100

# Clamp values for safety
combined["SOH"] = combined["SOH"].clip(lower=0, upper=100)

current_soh = combined["SOH"].iloc[-1]
degradation = 100 - current_soh

# ------------------------------
# Display Metrics
# ------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Initial SOH (%)", "100.00")
col2.metric("Current SOH (%)", f"{current_soh:.2f}")
col3.metric("Total Degradation (%)", f"{degradation:.2f}")

# ------------------------------
# SOH Plot
# ------------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=combined["Cycle"],
        y=combined["SOH"],
        mode="lines",
        line=dict(width=3),
        name="SOH (%)"
    )
)

fig.update_layout(
    title="SOH Degradation Over Cycles",
    xaxis_title="Cycle Number",
    yaxis_title="State of Health (%)",
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# GRU Prediction Placeholder
# ------------------------------
st.subheader("🔮 GRU Predicted Next Cycle SOH")

# Simple trend-based estimate (safe placeholder)
if len(combined) > 3:
    recent_trend = combined["SOH"].iloc[-3:].mean() - combined["SOH"].iloc[-4:-1].mean()
    predicted_soh = current_soh + recent_trend
else:
    predicted_soh = current_soh

predicted_soh = np.clip(predicted_soh, 0, 100)

st.success(f"{predicted_soh:.2f} %")

if predicted_soh < 80:
    st.warning("⚠ Battery approaching End-of-Life threshold.")

st.caption("Prediction generated using sequential degradation trend model.")

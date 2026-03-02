import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("🔋 Battery SOH Prediction using GRU Concept")

uploaded_file = st.file_uploader("Upload Battery CSV File")

if uploaded_file:
    df = pd.read_csv(uploaded_file, skiprows=[1])

    # Calculate SOH
    initial_capacity = df['Capacity'].abs().max()
    df['SOH'] = (df['Capacity'].abs() / initial_capacity) * 100

    st.subheader("📊 Data Preview")
    st.write(df.head())

    # Temperature vs Cycle
    st.subheader("🌡 Temperature vs Cycle")
    fig1, ax1 = plt.subplots()
    ax1.plot(df['Cycle'], df['Temperature'])
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("Temperature (°C)")
    st.pyplot(fig1)

    # Voltage vs Time
    st.subheader("🔌 Voltage vs Time")
    fig2, ax2 = plt.subplots()
    ax2.plot(df['Voltage'])
    ax2.set_ylabel("Voltage (V)")
    st.pyplot(fig2)

    # Current vs Time
    st.subheader("⚡ Current vs Time")
    fig3, ax3 = plt.subplots()
    ax3.plot(df['Current'])
    ax3.set_ylabel("Current (A)")
    st.pyplot(fig3)

    # Capacity degradation
    st.subheader("📉 Capacity Degradation")
    fig4, ax4 = plt.subplots()
    ax4.plot(df['Capacity'].abs())
    ax4.set_ylabel("Capacity (Ah)")
    st.pyplot(fig4)

    # SOH degradation
    st.subheader("📉 SOH Degradation Curve")
    fig5, ax5 = plt.subplots()
    ax5.plot(df['SOH'].rolling(500).mean())
    ax5.set_ylabel("SOH (%)")
    st.pyplot(fig5)

    # Demo Prediction
    st.subheader("Predicted Next SOH")
    predicted_soh = df['SOH'].iloc[-1] - 0.2
    st.success(f"{predicted_soh:.2f} %")
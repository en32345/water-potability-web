import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. SETTING HALAMAN WEBSITE
st.set_page_config(page_title="Water Quality App", page_icon="💧", layout="centered")

# 2. LOAD MODEL & SCALER
@st.cache_resource # Agar model tidak diload berulang kali (lebih cepat)
def load_model():
    model = pickle.load(open('model_air_terbaik.pkl', 'rb'))
    scaler = pickle.load(open('scaler_air.pkl', 'rb'))
    return model, scaler

try:
    model, scaler = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# 3. BAGIAN TAMPILAN HEADER
st.title("💧 Prediksi Kelayakan Air Minum")
st.write("Aplikasi ini menggunakan model **Random Forest** untuk menentukan apakah sampel air layak dikonsumsi.")
st.markdown("---")

# 4. MEMBUAT INPUTAN USER (SIDEBAR)
st.sidebar.header("📝 Masukkan Parameter Air")

def user_input():
    ph = st.sidebar.slider('pH (0-14)', 0.0, 14.0, 7.0)
    hardness = st.sidebar.number_input('Hardness (Kekerasan)', 47.0, 323.0, 150.0)
    solids = st.sidebar.number_input('Solids (Padatan)', 320.0, 61227.0, 20000.0)
    chloramines = st.sidebar.number_input('Chloramines', 0.3, 13.0, 7.0)
    sulfate = st.sidebar.number_input('Sulfate', 129.0, 481.0, 300.0)
    conductivity = st.sidebar.number_input('Conductivity', 181.0, 753.0, 400.0)
    organic_carbon = st.sidebar.number_input('Organic Carbon', 2.0, 28.0, 15.0)
    trihalomethanes = st.sidebar.number_input('Trihalomethanes', 0.7, 124.0, 60.0)
    turbidity = st.sidebar.number_input('Turbidity (Kekeruhan)', 1.4, 6.7, 4.0)
    
    data = {
        'ph': ph, 'Hardness': hardness, 'Solids': solids, 'Chloramines': chloramines,
        'Sulfate': sulfate, 'Conductivity': conductivity, 'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes, 'Turbidity': turbidity
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# TAMPILKAN DATA YANG DIINPUT
st.subheader("📊 Data Sampel Air Anda")
st.table(input_df)

# 5. TOMBOL PREDIKSI
if st.button("Analisis Kelayakan Air"):
    # Normalisasi input menggunakan scaler yang sama saat training
    input_scaled = scaler.transform(input_df)
    
    # Prediksi
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.markdown("---")
    st.subheader("🔍 Hasil Analisis Model:")
    
    if prediction[0] == 1:
        st.success("✅ **AIR LAYAK DIMINUM (POTABLE)**")
        st.balloons()
    else:
        st.error("❌ **AIR TIDAK LAYAK DIMINUM (NOT POTABLE)**")
        st.warning("Peringatan: Kandungan kimia melebihi ambang batas aman.")

    # Tampilkan Persentase Keyakinan
    confidence = np.max(prediction_proba) * 100
    st.write(f"Tingkat Keyakinan Model: **{confidence:.2f}%**")
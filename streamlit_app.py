import streamlit as st
import pickle
import numpy as np
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Kualitas Air", page_icon="💧")

# --- STYLE DESAIN (CSS) ---
st.markdown("""
<style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_model():
    try:
        # Membuka file model yang Anda upload ke GitHub
        with open("model_air.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler_air.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        return None, str(e)

model, scaler = load_model()

# --- TAMPILAN ---
st.title("🌊 Water Quality Analyzer")
st.write("Sistem Prediksi Kelayakan Air Minum (UAS Deep Learning)")

if model is None:
    st.error(f"File model/scaler tidak ditemukan! Pastikan file .pkl ada di GitHub. Error: {scaler}")
else:
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            ph = st.number_input("pH (0-14)", value=7.0, step=0.1)
            Hardness = st.number_input("Hardness", value=200.0)
            Solids = st.number_input("Solids", value=20000.0)
            Chloramines = st.number_input("Chloramines", value=7.0)
        with col2:
            Sulfate = st.number_input("Sulfate", value=300.0)
            Conductivity = st.number_input("Conductivity", value=400.0)
            Organic_carbon = st.number_input("Organic carbon", value=15.0)
            Trihalomethanes = st.number_input("Trihalomethanes", value=60.0)
        
        Turbidity = st.number_input("Turbidity", value=4.0)
        submitted = st.form_submit_button("PREDIKSI SEKARANG")

    if submitted:
        with st.spinner("Menganalisis data..."):
            time.sleep(1)
            # Data disusun sesuai urutan fitur saat training
            data = np.array([[ph, Hardness, Solids, Chloramines, Sulfate, 
                              Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
            
            # Proses Scaling dan Prediksi
            data_scaled = scaler.transform(data)
            prediction = model.predict(data_scaled)[0]
            
            st.markdown("---")
            if prediction == 1:
                st.balloons()
                st.success("### ✅ HASIL: AIR LAYAK MINUM (Potable)")
            else:
                st.error("### ❌ HASIL: AIR TIDAK LAYAK (Not Potable)")

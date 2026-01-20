import streamlit as st
import pickle
import numpy as np
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Water Quality Analyzer",
    page_icon="💧",
    layout="centered"
)

# --- STYLE CSS (Agar lebih menarik) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #007bff;
        color: white;
        height: 3em;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_model():
    # Pastikan nama file ini sama dengan yang ada di GitHub kamu
    with open("model_air.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler_air.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model()
except Exception as e:
    st.error(f"Gagal memuat model/scaler: {e}")

# --- HEADER ---
st.title("🌊 Sistem Prediksi Kelayakan Air")
st.write("Gunakan alat ini untuk menganalisis apakah air aman untuk dikonsumsi berdasarkan parameter laboratorium.")
st.markdown("---")

# --- INPUT PARAMETER ---
st.subheader("🧪 Input Parameter Air")
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        ph = st.number_input("pH (0-14)", value=7.0, step=0.1)
        hardness = st.number_input("Hardness (mg/L)", value=150.0)
        solids = st.number_input("Solids (ppm/TDS)", value=20000.0)
        chloramines = st.number_input("Chloramines (ppm)", value=7.0)
        sulfate = st.number_input("Sulfate (mg/L)", value=300.0)

    with col2:
        conductivity = st.number_input("Conductivity (μS/cm)", value=400.0)
        organic_carbon = st.number_input("Organic Carbon (ppm)", value=15.0)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=60.0)
        turbidity = st.number_input("Turbidity (NTU)", value=4.0)

# --- TOMBOL PREDIKSI ---
st.write("")
if st.button("ANALISIS SEKARANG"):
    with st.spinner('Sistem sedang melakukan pengecekan...'):
        time.sleep(1.2) # Efek simulasi analisis AI
        
        # Urutan harus sama dengan: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                conductivity, organic_carbon, trihalomethanes, turbidity]])
        
        # Transformasi dengan Scaler
        input_scaled = scaler.transform(input_data)
        
        # Prediksi
        prediction = model.predict(input_scaled)[0]

        # --- TAMPILAN HASIL ---
        st.markdown("### 📊 Hasil Analisis:")
        if prediction == 1:
            st.balloons()
            st.success("### ✅ LAYAK MINUM (Potable)")
            st.info("Air memenuhi kriteria kualitas untuk dikonsumsi berdasarkan data yang dimasukkan.")
        else:
            st.error("### ❌ TIDAK LAYAK MINUM (Not Potable)")
            st.warning("Peringatan: Kandungan air berisiko bagi kesehatan. Perlu proses filtrasi atau pengolahan lebih lanjut.")

# --- FOOTER ---
st.markdown("---")
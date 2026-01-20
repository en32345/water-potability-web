import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Kelayakan Air", layout="wide")

# Custom CSS untuk memperindah UI
st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("💧 Sistem Analisis & Prediksi Kelayakan Air")
st.write("Gunakan form di bawah untuk mengecek apakah air layak dikonsumsi.")

# --- LOAD & PREPROCESS DATA ---
@st.cache_resource
def train_model():
    df = pd.read_csv('water_potability.csv')
    # Imputasi Missing Values
    df['ph'] = df['ph'].fillna(df['ph'].mean())
    df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean())
    df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean())
    
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, df

model, accuracy, raw_data = train_model()

# --- INPUT FORM ---
st.subheader("📋 Masukkan Parameter Kualitas Air")
with st.container():
    col1, col2, col3 = st.columns([1, 1, 1.5]) # Kolom 3 lebih lebar untuk grafik
    
    with col1:
        ph = st.number_input("pH (Skala 0-14)", 0.0, 14.0, 7.0)
        hardness = st.number_input("Hardness (mg/L)", value=196.0)
        solids = st.number_input("Solids (ppm)", value=20000.0)
        chloramines = st.number_input("Chloramines (ppm)", value=7.0)
    
    with col2:
        sulfate = st.number_input("Sulfate (mg/L)", value=333.0)
        conductivity = st.number_input("Conductivity (μS/cm)", value=426.0)
        organic_carbon = st.number_input("Organic Carbon (ppm)", value=14.0)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=66.0)
        turbidity = st.number_input("Turbidity (NTU)", value=3.9)

    with col3:
        # --- VISUALISASI GAUGE pH ---
        st.write("**Visualisasi pH Air**")
        fig_ph = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = ph,
            title = {'text': "Indikator pH"},
            gauge = {
                'axis': {'range': [0, 14]},
                'bar': {'color': "#1f77b4"},
                'steps' : [
                    {'range': [0, 6.5], 'color': "#ff4b4b"}, # Asam
                    {'range': [6.5, 8.5], 'color': "#00cc96"}, # Ideal (WHO)
                    {'range': [8.5, 14], 'color': "#0068c9"}], # Basa
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': ph}
            }
        ))
        fig_ph.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_ph, use_container_width=True)

# --- PREDIKSI & HASIL ---
st.markdown("---")
if st.button("🚀 Analisis Kelayakan Sekarang", type="primary"):
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                            conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction[0] == 1:
            st.success("### ✅ LAYAK MINUM")
            st.write(f"Tingkat Kepercayaan: **{probability[0][1]*100:.1f}%**")
            # Menampilkan ilustrasi air bersih
            st.image("https://cdn-icons-png.flaticon.com/512/3105/3105807.png", width=150)
        else:
            st.error("### ❌ TIDAK LAYAK")
            st.write(f"Tingkat Kepercayaan: **{probability[0][0]*100:.1f}%**")
            # Menampilkan ilustrasi air tidak aman
            st.image("https://cdn-icons-png.flaticon.com/512/2844/2844260.png", width=150)

    with res_col2:
        st.write("#### Analisis Parameter:")
        # Logika sederhana untuk penjelasan pH
        if 6.5 <= ph <= 8.5:
            st.info("💡 **pH Normal:** Kadar keasaman air Anda sesuai standar kesehatan WHO.")
        else:
            st.warning("⚠️ **pH Tidak Ideal:** pH di luar rentang 6.5-8.5 dapat mempengaruhi rasa dan kesehatan.")
        
        if turbidity > 5.0:
            st.warning("⚠️ **Kekeruhan Tinggi:** Air terlihat keruh (Turbidity > 5 NTU), disarankan melalui penyaringan.")

# --- INFORMASI MODEL ---
with st.expander("ℹ️ Detail Akurasi Model"):
    st.write(f"Model: **Random Forest Classifier**")
    st.write(f"Akurasi: **{accuracy*100:.2f}%**")
    st.dataframe(raw_data.describe())
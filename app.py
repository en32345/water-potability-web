import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Water Quality AI", page_icon="💧", layout="wide")

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    stButton>button { width: 100%; border-radius: 20px; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <div style="background-color:#1e3d59; padding:20px; border-radius:15px; margin-bottom:25px">
    <h1 style="color:white; text-align:center; font-family:sans-serif;">💧 Smart Water Quality Analyzer</h1>
    <p style="color:#dcdde1; text-align:center;">Deteksi Kelayakan Air Minum Menggunakan Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("water_potability.csv")
        # Mengisi data kosong (NaN) dengan median masing-masing kolom agar tidak error
        df['ph'] = df['ph'].fillna(df['ph'].median())
        df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].median())
        df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median())
        return df
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")
        return None

df = load_and_clean_data()

if df is not None:
    # --- MODEL TRAINING ---
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- SIDEBAR NAVIGASI ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3105/3105807.png", width=100)
    st.sidebar.title("Menu Utama")
    menu = st.sidebar.radio("Pilih Menu:", ["📊 Dashboard Data", "🤖 Prediksi AI"])

    if menu == "📊 Dashboard Data":
        st.subheader("Analisis Statistik Dataset")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sampel", len(df))
        with col2:
            layak = len(df[df['Potability'] == 1])
            st.metric("Layak Minum (1)", layak)
        with col3:
            tidak_layak = len(df[df['Potability'] == 0])
            st.metric("Tidak Layak (0)", tidak_layak)

        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.write("**Distribusi pH Air**")
            fig1 = px.histogram(df, x="ph", color="Potability", marginal="box", color_discrete_sequence=['#e74c3c', '#2ecc71'])
            st.plotly_chart(fig1, use_container_width=True)
            
        with col_chart2:
            st.write("**Korelasi Parameter (Heatmap)**")
            corr = df.corr()
            fig2 = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig2, use_container_width=True)

    elif menu == "🤖 Prediksi AI":
        st.subheader("Cek Kelayakan Air")
        st.write("Silakan masukkan nilai parameter air untuk mendapatkan hasil prediksi:")

        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            
            with c1:
                ph = st.number_input("pH (0-14)", 0.0, 14.0, 7.0)
                hardness = st.number_input("Hardness", 0.0, 500.0, 200.0)
                solids = st.number_input("Solids (TDS)", 0.0, 100000.0, 20000.0)
            
            with c2:
                chloramines = st.number_input("Chloramines", 0.0, 20.0, 7.0)
                sulfate = st.number_input("Sulfate", 0.0, 1000.0, 300.0)
                conductivity = st.number_input("Conductivity", 0.0, 1000.0, 400.0)
                
            with c3:
                carbon = st.number_input("Organic Carbon", 0.0, 50.0, 15.0)
                triha = st.number_input("Trihalomethanes", 0.0, 200.0, 60.0)
                turbidity = st.number_input("Turbidity", 0.0, 10.0, 4.0)

            submit = st.form_submit_button("ANALISIS SEKARANG")

        if submit:
            data_input = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, carbon, triha, turbidity]])
            prediction = model.predict(data_input)
            
            st.markdown("### Hasil Analisis:")
            if prediction[0] == 1:
                st.balloons()
                st.success("✅ AIR DINYATAKAN LAYAK UNTUK DIMINUM")
                st.info("Keterangan: Berdasarkan algoritma, parameter air ini memenuhi standar keamanan.")
            else:
                st.error("❌ AIR TIDAK LAYAK UNTUK DIMINUM")
                st.warning("Keterangan: Parameter air berisiko bagi kesehatan, diperlukan pengolahan lebih lanjut.")

# --- FOOTER ---
st.markdown("---")
st.caption("Aplikasi ini dibuat untuk tujuan edukasi analisis data kualitas air.")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis pH Air", layout="wide")

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('water_potability.csv')
    # Membersihkan data pH yang kosong untuk analisis ini
    df_clean = df.dropna(subset=['ph'])
    return df_clean

df = load_data()

# Header
st.title("📊 Analisis Kelayakan pH Air")
st.markdown("Berapa nilai pH yang ideal berdasarkan dataset? Mari kita bedah datanya.")

# --- BAGIAN 1: RINGKASAN STATISTIK ---
col1, col2, col3 = st.columns(3)

avg_ph_potable = df[df['Potability'] == 1]['ph'].mean()
avg_ph_non_potable = df[df['Potability'] == 0]['ph'].mean()
standar_who = "6.5 - 8.5"

with col1:
    st.metric(label="Rerata pH Layak Minum", value=f"{avg_ph_potable:.2f}")
    st.caption("Berdasarkan data Potability = 1")

with col2:
    st.metric(label="Rerata pH Tidak Layak", value=f"{avg_ph_non_potable:.2f}")
    st.caption("Berdasarkan data Potability = 0")

with col3:
    st.metric(label="Standar Aman (WHO)", value=standar_who)
    st.caption("Standar Internasional")

st.divider()

# --- BAGIAN 2: VISUALISASI DISTRIBUSI ---
st.subheader("📈 Distribusi pH: Layak vs Tidak Layak")

col_left, col_right = st.columns([2, 1])

with col_left:
    # Grafik Histogram menggunakan Plotly
    fig = px.histogram(df, x="ph", color="Potability", 
                       marginal="box", # Menambahkan boxplot di atas
                       barmode="overlay",
                       color_discrete_map={0: "#EF553B", 1: "#00CC96"},
                       labels={"ph": "Derajat Keasaman (pH)", "Potability": "Layak?"},
                       title="Perbandingan Sebaran pH")
    fig.update_layout(legend_title_text='Keterangan (1=Layak)')
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.write("### Kesimpulan Data")
    st.info(f"""
    1. **pH Terendah (Layak):** {df[df['Potability']==1]['ph'].min():.2f}
    2. **pH Tertinggi (Layak):** {df[df['Potability']==1]['ph'].max():.2f}
    3. **Observasi:** Air yang layak minum dalam dataset ini terkonsentrasi kuat di angka **{avg_ph_potable:.1f}**.
    
    Meskipun ada air dengan pH ekstrim yang dilabeli layak, sebagian besar mengikuti pola normal mendekati netral (7).
    """)

# --- BAGIAN 3: CEK PH MANDIRI ---
st.divider()
st.subheader("🧪 Cek Status pH Anda")
user_ph = st.slider("Geser untuk menentukan angka pH:", 0.0, 14.0, 7.0)

if 6.5 <= user_ph <= 8.5:
    st.success(f"pH {user_ph} berada dalam rentang **IDEAL** (Standar WHO).")
else:
    if user_ph < 6.5:
        st.warning(f"pH {user_ph} cenderung **ASAM**. Perlu pengolahan lebih lanjut.")
    else:
        st.warning(f"pH {user_ph} cenderung **BASA**. Perlu pengolahan lebih lanjut.")

# Gauge Chart untuk Visualisasi Input
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = user_ph,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Indikator pH"},
    gauge = {
        'axis': {'range': [0, 14]},
        'bar': {'color': "black"},
        'steps' : [
            {'range': [0, 6.5], 'color': "red"},
            {'range': [6.5, 8.5], 'color': "green"},
            {'range': [8.5, 14], 'color': "blue"}],
        'threshold': {
            'line': {'color': "white", 'width': 4},
            'thickness': 0.75,
            'value': user_ph}
    }
))
st.plotly_chart(fig_gauge)
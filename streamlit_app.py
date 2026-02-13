import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Konoha Salary Predictor",
    page_icon="🍥",
    layout="wide"
)

# --- CUSTOM CSS (THEMA NARUTO) ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), 
                    url("https://images.alphacoders.com/134/1341995.png");
        background-size: cover;
    }
    .main-title {
        color: #E8691E;
        text-align: center;
        text-shadow: 2px 2px #203A43;
        font-weight: bold;
        font-size: 45px;
    }
    div.stButton > button:first-child {
        background-color: #E8691E !important;
        color: white !important;
        border-radius: 15px !important;
        font-weight: bold !important;
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('model_gb.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model_and_scaler()

if model is None:
    st.error("❌ File model_gb.pkl atau scaler.pkl tidak ditemukan!")
    st.stop()

# --- DATA CONFIG ---
feature_cols_final = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                       'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                       'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']

education_classes = np.array(['D3', 'S1', 'SMA', 'SMK'])
major_classes = np.array(['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])

# --- UI LAYOUT ---
st.markdown('<p class="main-title">🍥 Papan Misi Konoha: Prediksi Gaji</p>', unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2)

with col1:
    usia = st.slider('Usia', 18, 60, 25)
    pendidikan = st.selectbox('Pendidikan', education_classes)
    jenis_kelamin = st.radio('Jenis Kelamin', ['Laki-laki', 'Wanita'], horizontal=True)

with col2:
    jurusan = st.selectbox('Jurusan', major_classes)
    durasi_jam = st.number_input('Durasi Pelatihan (Jam)', 20, 1000, 60)
    nilai_ujian = st.slider('Nilai Ujian', 50.0, 100.0, 75.0)
    status_bekerja = st.radio('Status Bekerja', ['Belum Bekerja', 'Sudah Bekerja'], horizontal=True)

# --- PREPROCESSING ---
def preprocess_new_data(data):
    new_df = pd.DataFrame([data])
    
    # Label Encoding manual sesuai urutan training
    new_df['Pendidikan'] = np.where(education_classes == data['Pendidikan'])[0][0]
    new_df['Jurusan'] = np.where(major_classes == data['Jurusan'])[0][0]

    # One-Hot Encoding manual
    new_df['Jenis_Kelamin_Laki-laki'] = 1 if data['Jenis_Kelamin'] == 'Laki-laki' else 0
    new_df['Jenis_Kelamin_Wanita'] = 1 if data['Jenis_Kelamin'] == 'Wanita' else 0
    new_df['Status_Bekerja_Belum Bekerja'] = 1 if data['Status_Bekerja'] == 'Belum Bekerja' else 0
    new_df['Status_Bekerja_Sudah Bekerja'] = 1 if data['Status_Bekerja'] == 'Sudah Bekerja' else 0

    # Ambil kolom yang sesuai urutan
    final_input = new_df[feature_cols_final]
    return scaler.transform(final_input)

# --- TOMBOL PREDIKSI ---
if st.button('🔥 ANALISIS JALAN NINJAKU!'):
    input_data = {
        'Usia': usia, 'Durasi_Jam': durasi_jam, 'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan, 'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin, 'Status_Bekerja': status_bekerja
    }
    
    processed = preprocess_new_data(input_data)
    prediction = model.predict(processed)
    
    st.balloons()
    st.success(f"### 💰 Estimasi Gaji: {prediction[0]:.2f} Juta Rupiah")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Gaji - Edisi Ramadhan 1447 H",
    page_icon="🌙",
    layout="centered"
)

# --- CUSTOM CSS UNTUK TEMA RAMADHAN ---
st.markdown("""
    <style>
    /* Mengubah background utama */
    .stApp {
        background-color: #064e3b; /* Hijau Deep Islamic */
        color: #f3f4f6;
    }
    
    /* Mengubah warna Sidebar */
    [data-testid="stSidebar"] {
        background-color: #022c22;
    }

    /* Mengubah warna Header & Text */
    h1, h2, h3 {
        color: #fbbf24 !important; /* Warna Emas */
        font-family: 'Serif';
    }

    /* Tombol Prediksi */
    .stButton>button {
        background-color: #fbbf24;
        color: #064e3b;
        border-radius: 20px;
        font-weight: bold;
        border: none;
        width: 100%;
    }

    /* Input Fields */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the pre-trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    # Pastikan file ini ada di direktori yang sama
    with open('model_gb.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

model, scaler = load_model_and_scaler()

# --- HEADER APP ---
st.markdown("<h1 style='text-align: center;'>🌙 Barakah Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Menjemput Rezeki di Bulan Suci Ramadhan 1447 H</p>", unsafe_allow_html=True)
st.divider()

# User Inputs
st.header('🕌 Data Profil Peserta')

col1, col2 = st.columns(2)

with col1:
    usia = st.slider('Usia', 18, 60, 25)
    durasi_jam = st.slider('Durasi Pelatihan (Jam)', 20, 100, 60)
    nilai_ujian = st.slider('Nilai Ujian', 50.0, 100.0, 75.0, step=0.1)

with col2:
    pendidikan = st.selectbox('Pendidikan Terakhir', ['D3', 'S1', 'SMA', 'SMK'])
    jurusan = st.selectbox('Bidang Keahlian', ['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])
    jenis_kelamin = st.radio('Jenis Kelamin', ['Laki-laki', 'Wanita'])
    status_bekerja = st.radio('Status Saat Ini', ['Belum Bekerja', 'Sudah Bekerja'])

# (Fungsi Preprocessing tetap sama seperti kode asli Anda)
def preprocess_new_data(data):
    # ... (Gunakan logika preprocessing Anda yang asli di sini) ...
    # Saya ringkas agar fokus ke tema, pastikan logika di dalamnya tetap sama.
    new_df = pd.DataFrame([data])
    education_classes = np.array(['D3', 'S1', 'SMA', 'SMK'])
    major_classes = np.array(['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])
    
    new_df['Pendidikan'] = new_df['Pendidikan'].apply(lambda x: np.where(education_classes == x)[0][0] if x in education_classes else -1)
    new_df['Jurusan'] = new_df['Jurusan'].apply(lambda x: np.where(major_classes == x)[0][0] if x in major_classes else -1)

    feature_cols_final = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                          'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                          'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']
    
    for col in feature_cols_final[5:]: new_df[col] = 0
    if data['Jenis_Kelamin'] == 'Laki-laki': new_df['Jenis_Kelamin_Laki-laki'] = 1
    elif data['Jenis_Kelamin'] == 'Wanita': new_df['Jenis_Kelamin_Wanita'] = 1
    if data['Status_Bekerja'] == 'Belum Bekerja': new_df['Status_Bekerja_Belum Bekerja'] = 1
    elif data['Status_Bekerja'] == 'Sudah Bekerja': new_df['Status_Bekerja_Sudah Bekerja'] = 1

    new_df = new_df.drop(columns=['Jenis_Kelamin', 'Status_Bekerja'])
    preprocessed_input_scaled = scaler.transform(new_df[feature_cols_final])
    return pd.DataFrame(preprocessed_input_scaled, columns=feature_cols_final)

# --- ACTION BUTTON ---
if st.button('✨ Hitung Prediksi Gaji ✨'):
    new_data = {
        'Usia': usia, 'Durasi_Jam': durasi_jam, 'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan, 'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin, 'Status_Bekerja': status_bekerja
    }
    
    processed_data = preprocess_new_data(new_data)
    predicted_salary = model.predict(processed_data)

    st.balloons() # Efek perayaan
    st.markdown("---")
    st.subheader('Hasil Estimasi:')
    st.success(f'InsyaAllah, Estimasi Gaji Pertama: **{predicted_salary[0]:.2f} Juta Rupiah**')
    st.info("Semoga menjadi rezeki yang berkah dan bermanfaat.")

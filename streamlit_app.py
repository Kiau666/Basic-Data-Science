import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from sklearn.preprocessing import StandardScaler

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Gaji - Ramadhan 1447", page_icon="🌙")

# --- CSS KUSTOM (TEMA RAMADHAN) ---
def local_css():
    st.markdown(
        f"""
        <style>
        /* Mengimpor Font Google */
        @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Poppins:wght@300;400;600&display=swap');

        /* Kursor Unta Kustom */
        * {{
            cursor: url('https://img.icons8.com/ios-filled/30/FFD700/camel.png'), auto !important;
        }}

        /* Latar Belakang Utama (Gurun & Hijau) */
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), 
                        url('https://www.transparenttextures.com/patterns/sandpaper.png'),
                        #f4f1de;
            font-family: 'Poppins', sans-serif;
        }}

        /* Header & Judul */
        h1 {{
            color: #2E7D32 !important; /* Hijau Tua */
            font-family: 'Amiri', serif;
            text-align: center;
            text-shadow: 2px 2px #FFD700;
            font-size: 3rem !important;
        }}
        
        h2, h3 {{
            color: #1B5E20 !important;
            font-family: 'Amiri', serif;
        }}

        /* Kotak Input */
        .stSelectbox, .stSlider, .stRadio {{
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 15px;
            border: 2px solid #2E7D32;
            margin-bottom: 10px;
        }}

        /* Tombol Prediksi */
        div.stButton > button:first-child {{
            background-color: #2E7D32;
            color: #FFD700; /* Kuning Emas */
            border-radius: 20px;
            border: 2px solid #FFD700;
            width: 100%;
            font-weight: bold;
            font-size: 1.2rem;
            transition: 0.3s;
        }}

        div.stButton > button:first-child:hover {{
            background-color: #1B5E20;
            color: white;
            transform: scale(1.02);
        }}

        /* Hiasan Gurun Bawah */
        .desert-footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 100px;
            background-image: url('https://img.icons8.com/external-vitaliy-gorbachev-flat-vitaly-gorbachev/100/000000/external-desert-landscape-vitaliy-gorbachev-flat-vitaly-gorbachev.png');
            background-repeat: repeat-x;
            opacity: 0.2;
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# --- LOGIKA MODEL (SAMA SEPERTI SEBELUMNYA) ---
@st.cache_resource
def load_model_and_scaler():
    # Pastikan file ini ada di direktori Anda
    try:
        with open('model_gb.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("File model_gb.pkl atau scaler.pkl tidak ditemukan!")
        return None, None

model, scaler = load_model_and_scaler()

feature_cols_final = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                       'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                       'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']

education_classes = np.array(['D3', 'S1', 'SMA', 'SMK'])
major_classes = np.array(['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])

# --- TAMPILAN DEPAN ---
st.markdown("<h1>🌙 Marhaban ya Ramadhan 1447H 🌙</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Prediksi Gaji Pertama Peserta Vokasi</h3>", unsafe_allow_html=True)

# Ilustrasi Unta dan Gurun
col_mid1, col_mid2, col_mid3 = st.columns([1, 2, 1])
with col_mid2:
    st.image("https://img.icons8.com/external-flaticons-lineal-color-flat-icons/200/external-camel-ramadan-flaticons-lineal-color-flat-icons.png", width=200)

st.write("---")

# User Inputs
st.header('📝 Data Peserta Pelatihan')

col1, col2 = st.columns(2)

with col1:
    usia = st.slider('Usia', 18, 60, 25)
    durasi_jam = st.slider('Durasi Pelatihan (Jam)', 20, 100, 60)
    nilai_ujian = st.slider('Nilai Ujian', 50.0, 100.0, 75.0, step=0.1)

with col2:
    pendidikan = st.selectbox('Pendidikan Terakhir', ['D3', 'S1', 'SMA', 'SMK'])
    jurusan = st.selectbox('Jurusan Vokasi', ['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])
    jenis_kelamin = st.radio('Jenis Kelamin', ['Laki-laki', 'Wanita'])
    status_bekerja = st.radio('Status Saat Ini', ['Belum Bekerja', 'Sudah Bekerja'])

# Preprocessing Function
def preprocess_new_data(data):
    new_df = pd.DataFrame([data])

    def get_label_encoded_value(value, classes):
        if value in classes:
            return np.where(classes == value)[0][0]
        return -1

    new_df['Pendidikan'] = new_df['Pendidikan'].apply(lambda x: get_label_encoded_value(x, education_classes))
    new_df['Jurusan'] = new_df['Jurusan'].apply(lambda x: get_label_encoded_value(x, major_classes))

    one_hot_feature_cols = [
        'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
        'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
    ]

    for col in one_hot_feature_cols:
        new_df[col] = 0

    if data['Jenis_Kelamin'] == 'Laki-laki':
        new_df['Jenis_Kelamin_Laki-laki'] = 1
    elif data['Jenis_Kelamin'] == 'Wanita':
        new_df['Jenis_Kelamin_Wanita'] = 1

    if data['Status_Bekerja'] == 'Belum Bekerja':
        new_df['Status_Bekerja_Belum Bekerja'] = 1
    elif data['Status_Bekerja'] == 'Sudah Bekerja':
        new_df['Status_Bekerja_Sudah Bekerja'] = 1

    new_df = new_df.drop(columns=['Jenis_Kelamin', 'Status_Bekerja'])
    preprocessed_input_df = new_df[feature_cols_final]
    
    if scaler:
        preprocessed_input_scaled_array = scaler.transform(preprocessed_input_df)
        return pd.DataFrame(preprocessed_input_scaled_array, columns=feature_cols_final)
    return preprocessed_input_df

# Prediction Logic
if st.button('✨ Hitung Berkah Prediksi Gaji ✨'):
    if model and scaler:
        new_data = {
            'Usia': usia,
            'Durasi_Jam': durasi_jam,
            'Nilai_Ujian': nilai_ujian,
            'Pendidikan': pendidikan,
            'Jurusan': jurusan,
            'Jenis_Kelamin': jenis_kelamin,
            'Status_Bekerja': status_bekerja
        }

        processed_data = preprocess_new_data(new_data)
        predicted_salary = model.predict(processed_data)

        st.markdown("---")
        st.subheader('Hasil Prediksi:')
        st.success(f'Estimasi Gaji Pertama: **{predicted_salary[0]:.2f} Juta Rupiah**')
        st.balloons()
    else:
        st.error("Model tidak tersedia. Pastikan file .pkl sudah diunggah.")

# Footer visual tambahan
st.markdown('<div class="desert-footer"></div>', unsafe_allow_html=True)
st.caption("Aplikasi Prediksi Gaji Vokasi - Edisi Ramadhan 1447H")

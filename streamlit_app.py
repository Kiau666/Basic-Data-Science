import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Gaji - Ramadhan 1447 H",
    page_icon="🌙",
    layout="centered"
)

# --- 2. CUSTOM CSS (TEMA RAMADHAN & HACK KETUPAT) ---
st.markdown("""
    <style>
    /* Background Utama */
    .stApp {
        background-color: #064e3b; 
        color: #f3f4f6;
    }
    
    /* Header & Teks */
    h1, h2, h3, p {
        color: #fbbf24 !important;
        font-family: 'Georgia', serif;
    }

    /* Tombol Hitung Berkah Gaji */
    .stButton>button {
        background-color: #fbbf24;
        color: #064e3b;
        border-radius: 25px;
        font-weight: bold;
        font-size: 20px;
        border: 2px solid #fcd34d;
        width: 100%;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #fcd34d;
        color: #064e3b;
        transform: scale(1.02);
    }

    /* Custom Efek Salju Menjadi Ketupat */
    [data-testid="stSnow"] {
        background-image: url("https://cdn-icons-png.flaticon.com/512/3520/3520844.png");
        background-size: 60px;
        opacity: 0.8;
    }

    /* Styling Input Box */
    .stNumberInput, .stSelectbox, .stSlider, .stRadio {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD MODEL & SCALER ---
@st.cache_resource
def load_model_and_scaler():
    # Pastikan file .pkl ada di folder yang sama dengan script ini
    with open('model_gb.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except FileNotFoundError:
    st.error("⚠️ File model_gb.pkl atau scaler.pkl tidak ditemukan!")

# Definisi kolom fitur
feature_cols_final = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                       'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                       'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']

education_classes = np.array(['D3', 'S1', 'SMA', 'SMK'])
major_classes = np.array(['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])

# --- 4. HEADER APLIKASI ---
st.markdown("<h1 style='text-align: center;'>🌙 Barakah Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Edisi Spesial Ramadhan 1447 H - Menjemput Rezeki Berkah</p>", unsafe_allow_html=True)
st.divider()

# --- 5. INPUT USER ---
st.header('🕌 Data Profil Peserta')

col1, col2 = st.columns(2)

with col1:
    usia = st.slider('Usia Anda', 18, 60, 25)
    durasi_jam = st.slider('Durasi Pelatihan (Jam)', 20, 100, 60)
    nilai_ujian = st.slider('Nilai Hasil Ujian', 50.0, 100.0, 75.0, step=0.1)

with col2:
    pendidikan = st.selectbox('Pendidikan Terakhir', education_classes)
    jurusan = st.selectbox('Jurusan Vokasi', major_classes)
    jenis_kelamin = st.radio('Jenis Kelamin', ['Laki-laki', 'Wanita'])
    status_bekerja = st.radio('Status Pekerjaan', ['Belum Bekerja', 'Sudah Bekerja'])

# --- 6. FUNGSI PREPROCESSING ---
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
    
    # Scaling
    preprocessed_input_scaled_array = scaler.transform(preprocessed_input_df)
    return pd.DataFrame(preprocessed_input_scaled_array, columns=feature_cols_final)

# --- 7. TOMBOL PREDIKSI & EFEK ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button('✨ Hitung Berkah Gaji ✨'):
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

    # Efek 1: Ketupat (Snow Hack)
    st.snow()

    # Efek 2: Kembang Api (JS Confetti)
    st.components.v1.html("""
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
        <script>
            var end = Date.now() + (4 * 1000);
            var colors = ['#fbbf24', '#ffffff', '#22c55e'];

            (function frame() {
              confetti({
                particleCount: 3,
                angle: 60,
                spread: 55,
                origin: { x: 0 },
                colors: colors
              });
              confetti({
                particleCount: 3,
                angle: 120,
                spread: 55,
                origin: { x: 1 },
                colors: colors
              });

              if (Date.now() < end) {
                requestAnimationFrame(frame);
              }
            }());
        </script>
    """, height=0)

    # --- HASIL AKHIR ---
    st.markdown("---")
    st.subheader('Hasil Prediksi Rezeki:')
    st.success(f'💸 Estimasi Gaji Pertama: **{predicted_salary[0]:.2f} Juta Rupiah**')
    st.info("💡 *Tips: Jangan lupa zakat dan sedekah agar penghasilan semakin berkah.*")

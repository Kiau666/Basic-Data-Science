import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Gaji Ramadhan 1447H",
    page_icon="🌙",
    layout="centered"
)

# --- 2. CUSTOM CSS (TEMA RAMADHAN & WARNA TULISAN) ---
st.markdown("""
    <style>
    /* Background utama hijau tua keislaman */
    .stApp {
        background-color: #064e3b;
    }
    
    /* Mengubah semua teks utama menjadi Putih Tulang/Emas agar terbaca */
    .stApp, p, span, label {
        color: #fef3c7 !important; /* Cream/Off-white */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Judul dan Subheader berwarna Emas Menyala */
    h1, h2, h3 {
        color: #fbbf24 !important; /* Gold */
        text-shadow: 2px 2px 4px #000000;
    }

    /* Tombol Hitung Berkah Gaji */
    .stButton>button {
        background-color: #fbbf24 !important;
        color: #064e3b !important;
        border-radius: 30px !important;
        font-weight: bold !important;
        font-size: 22px !important;
        height: 3em !important;
        width: 100% !important;
        border: 2px solid #fcd34d !important;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    }
    
    .stButton>button:hover {
        background-color: #fcd34d !important;
        transform: scale(1.01);
        transition: 0.2s;
    }

    /* Efek Salju diubah menjadi Ketupat */
    [data-testid="stSnow"] {
        background-image: url("https://cdn-icons-png.flaticon.com/512/3520/3520844.png");
        background-size: 50px;
    }

    /* Mempercantik Input Fields */
    .stSelectbox, .stSlider, .stRadio {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 10px;
        border: 1px solid #fbbf24;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD MODEL & SCALER ---
@st.cache_resource
def load_model_and_scaler():
    # Pastikan file pkl Anda tersedia di direktori yang sama
    with open('model_gb.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

# Load data model
try:
    model, scaler = load_model_and_scaler()
except:
    st.warning("⚠️ File model_gb.pkl atau scaler.pkl belum terdeteksi. Pastikan file tersedia.")

# Variabel pendukung
feature_cols_final = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                       'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                       'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']
education_classes = np.array(['D3', 'S1', 'SMA', 'SMK'])
major_classes = np.array(['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])

# --- 4. TAMPILAN HEADER ---
st.markdown("<h1 style='text-align: center;'>🌙 Prediksi Gaji Berkah</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Ramadhan Karim 1447 H</h3>", unsafe_allow_html=True)
st.write("---")

# --- 5. INPUT DATA (USER INTERFACE) ---
st.subheader('🕌 Data Peserta Pelatihan')

col1, col2 = st.columns(2)

with col1:
    usia = st.slider('Usia', 18, 60, 25)
    durasi_jam = st.slider('Durasi Pelatihan (Jam)', 20, 100, 60)
    nilai_ujian = st.slider('Nilai Ujian', 50.0, 100.0, 75.0, step=0.1)

with col2:
    pendidikan = st.selectbox('Pendidikan', education_classes)
    jurusan = st.selectbox('Jurusan', major_classes)
    jenis_kelamin = st.radio('Jenis Kelamin', ['Laki-laki', 'Wanita'])
    status_bekerja = st.radio('Status Bekerja', ['Belum Bekerja', 'Sudah Bekerja'])

# --- 6. FUNGSI PREPROCESSING ---
def preprocess_new_data(data):
    new_df = pd.DataFrame([data])
    
    def get_label_encoded_value(value, classes):
        if value in classes:
            return np.where(classes == value)[0][0]
        return -1

    new_df['Pendidikan'] = new_df['Pendidikan'].apply(lambda x: get_label_encoded_value(x, education_classes))
    new_df['Jurusan'] = new_df['Jurusan'].apply(lambda x: get_label_encoded_value(x, major_classes))

    for col in ['Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita', 'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']:
        new_df[col] = 0

    if data['Jenis_Kelamin'] == 'Laki-laki': new_df['Jenis_Kelamin_Laki-laki'] = 1
    elif data['Jenis_Kelamin'] == 'Wanita': new_df['Jenis_Kelamin_Wanita'] = 1

    if data['Status_Bekerja'] == 'Belum Bekerja': new_df['Status_Bekerja_Belum Bekerja'] = 1
    elif data['Status_Bekerja'] == 'Sudah Bekerja': new_df['Status_Bekerja_Sudah Bekerja'] = 1

    new_df = new_df.drop(columns=['Jenis_Kelamin', 'Status_Bekerja'])
    preprocessed_input_df = new_df[feature_cols_final]
    preprocessed_input_scaled = scaler.transform(preprocessed_input_df)
    
    return pd.DataFrame(preprocessed_input_scaled, columns=feature_cols_final)

# --- 7. TOMBOL DAN EFEK PERAYAAN ---
st.write("")
if st.button('✨ Hitung Berkah Gaji ✨'):
    # Jalankan Animasi Ketupat (st.snow yang sudah di-custom CSS)
    st.snow()
    
    # Jalankan Animasi Kembang Api (JavaScript Confetti)
    st.components.v1.html("""
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
        <script>
            var end = Date.now() + (4 * 1000);
            var colors = ['#fbbf24', '#ffffff', '#22c55e', '#fcd34d'];

            (function frame() {
              confetti({
                particleCount: 5,
                angle: 60,
                spread: 55,
                origin: { x: 0, y: 0.7 },
                colors: colors
              });
              confetti({
                particleCount: 5,
                angle: 120,
                spread: 55,
                origin: { x: 1, y: 0.7 },
                colors: colors
              });

              if (Date.now() < end) {
                requestAnimationFrame(frame);
              }
            }());
        </script>
    """, height=0)

    # Logika Prediksi
    input_data = {
        'Usia': usia, 'Durasi_Jam': durasi_jam, 'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan, 'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin, 'Status_Bekerja': status_bekerja
    }

    processed_data = preprocess_new_data(input_data)
    predicted_salary = model.predict(processed_data)

    # Menampilkan Hasil
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Hasil Estimasi</h2>", unsafe_allow_html=True)
    st.success(f"Masya Allah, Gaji Pertama Anda Diprediksi: {predicted_salary[0]:.2f} Juta Rupiah")
    st.info("Semoga rezeki Anda senantiasa berkah dan melimpah di bulan suci ini.")

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
    /* Background & Font */
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)), 
                    url("https://images.alphacoders.com/134/1341995.png");
        background-size: cover;
    }
    
    /* Title Styling */
    .main-title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #E8691E; /* Naruto Orange */
        text-align: center;
        text-shadow: 2px 2px #203A43;
        font-weight: bold;
        font-size: 50px;
    }

    /* Card Styling */
    div.stButton > button:first-child {
        background-color: #E8691E;
        color: white;
        border-radius: 20px;
        border: 2px solid #203A43;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #203A43;
        color: #E8691E;
        border: 2px solid #E8691E;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #203A43;
    }

    /* Success Box */
    .stSuccess {
        background-color: rgba(232, 105, 30, 0.2);
        border: 1px solid #E8691E;
        color: #203A43;
    }
    </style>
    """, unsafe_state=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model_and_scaler():
    # Pastikan file ini ada di direktori yang sama
    with open('model_gb.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except FileNotFoundError:
    st.error("⚠️ File model_gb.pkl atau scaler.pkl tidak ditemukan! Pastikan file model sudah diupload.")

# --- DATA CONFIG ---
feature_cols_final = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                       'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                       'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']

education_classes = np.array(['D3', 'S1', 'SMA', 'SMK'])
major_classes = np.array(['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])

# --- UI LAYOUT ---
st.markdown('<p class="main-title">🍥 Papan Misi Konoha: Prediksi Ryo</p>', unsafe_allow_html=True)
st.write("<p style='text-align: center;'>Tentukan masa depan Shinobi-mu! Masukkan data untuk memprediksi gaji pertama (Ryo).</p>", unsafe_allow_html=True)
st.divider()

# Gunakan kolom untuk layout yang lebih rapi
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("👤 Profil Shinobi")
    usia = st.slider('Usia (Tahun)', 18, 60, 25)
    pendidikan = st.selectbox('Tingkat Akademi (Pendidikan)', education_classes)
    jenis_kelamin = st.radio('Jenis Kelamin', ['Laki-laki', 'Wanita'], horizontal=True)
    status_bekerja = st.radio('Status Saat Ini', ['Belum Bekerja', 'Sudah Bekerja'], horizontal=True)

with col2:
    st.subheader("📜 Spesialisasi Jutsu")
    jurusan = st.selectbox('Bidang Keahlian (Jurusan)', major_classes)
    durasi_jam = st.number_input('Total Jam Latihan (Durasi)', 20, 1000, 60)
    nilai_ujian = st.slider('Skor Ujian Akhir', 50.0, 100.0, 75.0, step=0.1)

# --- PREPROCESSING FUNCTION ---
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

# --- PREDICTION LOGIC ---
st.write("") # Spacer
if st.button('🔥 ANALISIS JALAN NINJAKU!'):
    new_data = {
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Bekerja': status_bekerja
    }

    with st.spinner('Menghitung Chakra...'):
        processed_data = preprocess_new_data(new_data)
        predicted_salary = model.predict(processed_data)

    st.balloons()
    st.markdown("---")
    st.subheader('💰 Hasil Estimasi Tunjangan Misi:')
    st.success(f'### **{predicted_salary[0]:.2f} Juta Ryo (Rupiah)**')
    st.info("Ingat! Ini hanyalah prediksi. Teruslah berlatih seperti Lee agar hasilnya melampaui ekspektasi! 👊")

# --- FOOTER ---
st.markdown("<br><br><p style='text-align: center; color: grey;'>Dibuat dengan semangat api Konoha 🔥</p>", unsafe_allow_html=True)

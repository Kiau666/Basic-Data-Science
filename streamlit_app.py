import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --- CUSTOM CSS UNTUK TEMA TAHUN BARU CINA ---
st.set_page_config(page_title="Prediksi Gaji Imlek", page_icon="🧧")

st.markdown("""
    <style>
    /* Mengatur latar belakang utama */
    .stApp {
        background-color: #8B0000; /* Merah Tua */
        color: #FFD700; /* Emas */
    }
    
    /* Mengatur warna sidebar jika ada */
    [data-testid="stSidebar"] {
        background-color: #5C0000;
    }

    /* Mengubah warna teks judul dan header */
    h1, h2, h3, p, span, label {
        color: #FFD700 !important;
        font-family: 'Trebuchet MS', sans-serif;
    }

    /* Kustomisasi tombol */
    .stButton>button {
        background-color: #FFD700;
        color: #8B0000;
        border-radius: 20px;
        border: 2px solid #FFA500;
        font-weight: bold;
        width: 100%;
        height: 3em;
    }
    
    .stButton>button:hover {
        background-color: #FFA500;
        color: white;
    }

    /* Input styling */
    .stSlider, .stSelectbox, .stRadio {
        background-color: rgba(255, 215, 0, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #FFD700;
    }

    /* Animasi Lampion Sederhana (Opsional) */
    .lampion-text {
        text-align: center;
        font-size: 50px;
        margin-bottom: 0px;
    }
    </style>
    """, unsafe_allow_html=True)

# Tambah dekorasi Header
st.markdown("<div class='lampion-text'>🏮 🐉 🏮</div>", unsafe_allow_html=True)

# --- KODE ASLI (TIDAK DIUBAH) ---

# Load the pre-trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    with open('model_gb.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

model, scaler = load_model_and_scaler()

# Define the feature columns and their order used during training
feature_cols_final = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                       'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                       'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']

# Define the possible categories for categorical features (used for Label and One-Hot Encoding)
education_classes = np.array(['SMA', 'SMK', 'D3', 'S1', 'S2'])
major_classes = np.array(['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])

# --- UI MODIFIKASI TEMA ---
st.title('🧧 Prediksi Keberuntungan Gaji Pertama')
st.write('Rayakan Tahun Baru Cina dengan melihat potensi kemakmuran finansial Anda dari hasil pelatihan vokasi.')

# User Inputs
st.header('🧧 Data Keberuntungan Peserta')

usia = st.slider('Usia', 18, 60, 25)
durasi_jam = st.slider('Durasi Pelatihan (Jam)', 20, 100, 60)
nilai_ujian = st.slider('Nilai Ujian', 50.0, 100.0, 75.0, step=0.1)
pendidikan = st.selectbox('Tingkat Pendidikan', ['SMA', 'SMK', 'D3', 'S1', 'S2'])
jurusan = st.selectbox('Bidang Keahlian (Jurusan)', ['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])
jenis_kelamin = st.radio('Jenis Kelamin', ['Laki-laki', 'Wanita'])
status_bekerja = st.radio('Status Karir Saat Ini', ['Belum Bekerja', 'Sudah Bekerja'])

# Preprocessing Function for new data
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

    preprocessed_input_scaled_array = scaler.transform(preprocessed_input_df)
    preprocessed_input = pd.DataFrame(preprocessed_input_scaled_array, columns=feature_cols_final)

    return preprocessed_input


if st.button('🧧 Prediksi Kemakmuran (Gaji)'):
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

    st.subheader('Estimasi Berkah Gaji Anda:')
    # Menampilkan hasil dengan kotak sukses bertema emas/merah
    st.balloons()
    st.success(f'恭喜发财 (Gong Xi Fa Cai)! Gaji Pertama yang Diprediksi: Rp {predicted_salary[0]:.2f} Juta Rupiah')

st.markdown("---")
st.markdown("<p style='text-align: center;'>Semoga Tahun Ini Membawa Keberuntungan dan Kesuksesan! 🧧🐉</p>", unsafe_allow_html=True)

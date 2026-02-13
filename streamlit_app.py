
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

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

# Streamlit App Title
st.title('Prediksi Gaji Pertama Peserta Pelatihan Vokasi')
st.write('Aplikasi ini memprediksi gaji pertama berdasarkan data peserta pelatihan.')

# User Inputs
st.header('Data Peserta')

usia = st.slider('Usia', 18, 60, 25)
durasi_jam = st.slider('Durasi Pelatihan (Jam)', 20, 100, 60)
nilai_ujian = st.slider('Nilai Ujian', 50.0, 100.0, 75.0, step=0.1)
pendidikan = st.selectbox('Pendidikan', ['SMA', 'SMK', 'D3', 'S1', 'S2']
jurusan = st.selectbox('Jurusan', ['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])
jenis_kelamin = st.radio('Jenis Kelamin', ['Laki-laki', 'Wanita'])
status_bekerja = st.radio('Status Bekerja', ['Belum Bekerja', 'Sudah Bekerja'])

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


if st.button('Prediksi Gaji Pertama'):
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

    st.subheader('Hasil Prediksi:')
    st.success(f'Gaji Pertama yang Diprediksi: {predicted_salary[0]:.2f} Juta Rupiah')

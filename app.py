import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Memuat model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Prediksi Harga Laptop")

# Merek
company = st.selectbox('Merek', df['Company'].unique())

# Tipe laptop
type = st.selectbox('Tipe', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (dalam GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Berat
weight = st.number_input('Berat Laptop', min_value=0.0)

# Layar Sentuh
touchscreen = st.selectbox('Layar Sentuh', ['Tidak', 'Ya'])

# IPS
ips = st.selectbox('IPS', ['Tidak', 'Ya'])

# Ukuran layar
screen_size = st.number_input('Ukuran Layar', min_value=1.0)

# Resolusi layar
resolution = st.selectbox('Resolusi Layar', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD (dalam GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (dalam GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS
os = st.selectbox('OS', df['os'].unique())

if st.button('Prediksi Harga'):
    # Konversi input ke format yang benar
    touchscreen = 1 if touchscreen == 'Ya' else 0
    ips = 1 if ips == 'Ya' else 0

    # Validasi ukuran layar
    if screen_size <= 0:
        st.error("Ukuran layar harus lebih besar dari nol.")
    else:
        try:
            # Hitung PPI
            x_res, y_res = map(int, resolution.split('x'))
            ppi = ((x_res**2 + y_res**2)**0.5) / screen_size

            # Siapkan DataFrame query
            query = pd.DataFrame({
                'Company': [company],
                'TypeName': [type],
                'Ram': [ram],
                'Weight': [weight],
                'Touchscreen': [touchscreen],
                'Ips': [ips],
                'Cpu brand': [cpu],
                'HDD': [hdd],
                'SSD': [ssd],
                'Gpu brand': [gpu],
                'os': [os],
                'ppi': [ppi]
            })

            # Prediksi harga
            predicted_price_usd = np.exp(pipe.predict(query))[0]
            exchange_rate = 14500  # Kurs tetap dari USD ke IDR
            predicted_price_idr = predicted_price_usd * exchange_rate
            st.title(f"Untuk harga laptop yang anda cari adalah Rp : {int(predicted_price_idr):,} ")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
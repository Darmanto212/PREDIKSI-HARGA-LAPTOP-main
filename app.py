import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Memuat model dan dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Prediksi Harga Laptop")

# Input pengguna
company = st.selectbox('Merek', df['Company'].unique())
type = st.selectbox('Tipe', df['TypeName'].unique())
ram = st.selectbox('RAM (dalam GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Berat Laptop', min_value=0.0)
touchscreen = st.selectbox('Layar Sentuh', ['Tidak', 'Ya'])
ips = st.selectbox('IPS', ['Tidak', 'Ya'])
screen_size = st.number_input('Ukuran Layar', min_value=1.0)
resolution = st.selectbox('Resolusi Layar', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (dalam GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (dalam GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

# Tombol prediksi
if st.button('Prediksi Harga'):
    # Konversi input ke format numerik
    touchscreen = 1 if touchscreen == 'Ya' else 0
    ips = 1 if ips == 'Ya' else 0

    if screen_size <= 0:
        st.error("Ukuran layar harus lebih besar dari nol.")
    else:
        try:
            # Hitung PPI
            x_res, y_res = map(int, resolution.split('x'))
            ppi = ((x_res**2 + y_res**2)**0.5) / screen_size

            # Siapkan input model
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

            # Prediksi
            predicted_price_usd = np.exp(pipe.predict(query))[0]
            exchange_rate = 14500  # Kurs tetap USD ke IDR
            predicted_price_idr = predicted_price_usd * exchange_rate

            st.title(f"Untuk harga laptop yang anda cari adalah Rp : {int(predicted_price_idr):,} ")

            # ---------------------
            # Tambahan: Grafik RAM
            # ---------------------
            st.subheader("Grafik Prediksi Harga terhadap RAM")

            ram_variasi = [2, 4, 6, 8, 12, 16, 24, 32, 64]
            harga_prediksi = []

            for r in ram_variasi:
                q = query.copy()
                q['Ram'] = r
                pred = np.exp(pipe.predict(q))[0] * exchange_rate
                harga_prediksi.append(pred)

            df_grafik = pd.DataFrame({
                'RAM (GB)': ram_variasi,
                'Harga (Rp)': harga_prediksi
            })

            fig, ax = plt.subplots()
            ax.plot(df_grafik['RAM (GB)'], df_grafik['Harga (Rp)'], marker='o', color='blue')
            ax.set_xlabel("RAM (GB)")
            ax.set_ylabel("Harga (Rp)")
            ax.set_title("Pengaruh RAM terhadap Harga Laptop")
            ax.grid(True)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

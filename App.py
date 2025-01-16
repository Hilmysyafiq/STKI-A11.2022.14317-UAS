import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Fungsi untuk memuat dataset
def load_data():
    # Dataset diharapkan ada di direktori yang sama
    file_path = "penjualan_sepeda_motor_bekas.csv"
    data = pd.read_csv(file_path)
    return data

# Judul aplikasi
st.title("Prediksi Harga Sepeda Motor Bekas")

try:
    # Memuat dataset
    data = load_data()

    # Mengisi nilai yang hilang dengan median
    numerical_columns = data.select_dtypes(include=['number']).columns
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

    # Encoding variabel kategorikal
    data = pd.get_dummies(data, drop_first=True)

    # Pisahkan fitur dan target
    try:
        X = data.drop('harga', axis=1)
        y = data['harga']
    except KeyError:
        st.error("Dataset tidak memiliki kolom 'harga'.")
    else:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("**Evaluasi Model:**")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"R-squared Score: {r2:.2f}")

        # Form input untuk prediksi
        st.write("## Masukkan Informasi Sepeda Motor")

        # Input pengguna untuk model, jenis, dan transmisi
        model_options = [col.replace("model_", "") for col in X.columns if col.startswith("model_")]
        jenis_options = [col.replace("jenis_", "") for col in X.columns if col.startswith("jenis_")]
        transmisi_options = [col.replace("transmisi_", "") for col in X.columns if col.startswith("transmisi_")]

        if "Automatic" not in transmisi_options:
            transmisi_options.append("Automatic")  # Tambahkan opsi "Automatic" jika tidak ada

        selected_model = st.selectbox("Pilih Model", model_options, key="model_select")
        selected_jenis = st.selectbox("Pilih Jenis", jenis_options, key="jenis_select")
        selected_transmisi = st.selectbox("Pilih Transmisi", transmisi_options, key="transmisi_select")

        # Variabel input pengguna lainnya
        user_inputs = {}
        user_inputs[f"model_{selected_model}"] = 1
        user_inputs[f"jenis_{selected_jenis}"] = 1
        user_inputs[f"transmisi_{selected_transmisi}"] = 1

        for column in X.columns:
            if not (column.startswith("model_") or column.startswith("jenis_") or column.startswith("transmisi_") or column in ["pajak", "konsumsiBBM", "odometer"]):
                # Mengganti label 'odometer' menjadi 'kilometer yang ditempuh'
                label = column
                if "mesin" in column.lower():
                    label = "Mesin (CC Motor)"
                user_inputs[column] = st.number_input(f"Masukkan {label}", value=0, step=1, format="%d")

        # Melengkapi data input dengan nilai 0 untuk kolom yang tidak dipilih
        input_data = pd.DataFrame([user_inputs]).reindex(columns=X.columns, fill_value=0)

        if st.button("Prediksi Harga"):
            # Prediksi harga
            prediksi = model.predict(input_data)[0]
            st.success(f"Harga yang diprediksi: Rp {prediksi:,.2f}")

except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")

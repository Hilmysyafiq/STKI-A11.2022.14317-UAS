import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Fungsi untuk memuat dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Judul aplikasi
st.title("Prediksi Harga Sepeda Motor Bekas")

# Input path file lokal secara manual
file_path = st.text_input("Masukkan path file CSV lokal:", value="penjualan_sepeda_motor_bekas.csv")

if file_path:
    try:
        # Memuat dataset dari file lokal
        data = load_data(file_path)

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

            # Prediksi
            y_pred = model.predict(X_test)

            # Evaluasi
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write("**Evaluasi Model:**")
            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
            st.write(f"R-squared Score: {r2:.2f}")

            # Form input untuk prediksi
            st.write("## Prediksi Harga Sepeda Motor")
            inputs = {}
            for column in X.columns:
                inputs[column] = st.number_input(f"Masukkan nilai untuk {column}", value=0.0)

            if st.button("Prediksi Harga"):
                input_data = np.array([list(inputs.values())]).reshape(1, -1)
                prediksi = model.predict(input_data)[0]
                st.success(f"Harga yang diprediksi: Rp {prediksi:,.2f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")

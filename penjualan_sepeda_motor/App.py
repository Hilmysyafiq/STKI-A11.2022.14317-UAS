import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Judul aplikasi
st.title('Analisis Penjualan Sepeda Motor Bekas')

# Upload dataset
uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Data yang diunggah:")
    st.write(data.head())

    # Informasi dataset
    st.subheader('Informasi Dataset')
    st.write(data.info())

    # Statistik deskriptif
    st.subheader('Statistik Deskriptif')
    st.write(data.describe())

    # Mengecek nilai yang hilang
    missing_data = data.isnull().sum()
    st.subheader('Nilai yang Hilang')
    st.write(missing_data)

    # Visualisasi nilai yang hilang
    st.subheader('Visualisasi Nilai yang Hilang')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis', ax=ax)
    st.pyplot(fig)

    # Distribusi harga
    if 'harga' in data.columns:
        st.subheader('Distribusi Harga Sepeda Motor Bekas')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['harga'], bins=20, kde=True, color='blue', ax=ax)
        st.pyplot(fig)

    # Heatmap korelasi
    data_numerik = data.select_dtypes(include=['float64', 'int64'])
    st.subheader('Heatmap Korelasi Antar Variabel Numerik')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data_numerik.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    # Scatterplot Tahun vs Harga
    if 'tahun' in data.columns and 'harga' in data.columns:
        st.subheader('Hubungan Tahun Produksi dan Harga')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=data['tahun'], y=data['harga'], alpha=0.7, color='green', ax=ax)
        st.pyplot(fig)

    # Mengisi nilai yang hilang
    numerical_columns = data.select_dtypes(include=['number']).columns
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

    # Encoding variabel kategorikal
    data = pd.get_dummies(data, drop_first=True)

    # Pemisahan fitur dan target
    try:
        X = data.drop('harga', axis=1)
        y = data['harga']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Membuat model regresi linear
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi model
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader('Evaluasi Model')
        st.write(f'Mean Absolute Error (MAE): {mae}')
        st.write(f'R-squared Score: {r2}')

    except KeyError as e:
        st.error(f"Error: {e}")
        st.write("Kolom yang tersedia dalam dataset:", data.columns)
else:
    st.info("Silakan unggah file CSV untuk memulai analisis.")

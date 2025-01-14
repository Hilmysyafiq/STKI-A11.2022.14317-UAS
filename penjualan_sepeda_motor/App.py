from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load model dan fitur kolom
model = pickle.load(open('model.pkl', 'rb'))  # Simpan model Anda ke dalam file 'model.pkl'
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))  # Simpan daftar kolom fitur ke dalam file 'feature_columns.pkl'

@app.route('/')
def home():
    return render_template('index.html')  # Halaman HTML untuk input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data input dari form HTML
        input_data = request.form
        
        # Konversi input ke DataFrame
        input_df = pd.DataFrame([{col: input_data[col] for col in feature_columns}])
        
        # Pastikan tipe data sesuai dengan model
        input_df = input_df.astype(float)

        # Prediksi menggunakan model
        prediction = model.predict(input_df)[0]

        return jsonify({
            'prediction': f'Rp {prediction:,.2f}'
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)

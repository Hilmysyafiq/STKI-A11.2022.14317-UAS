import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump, load

# Load dataset
file_path = 'penjualan_sepeda_motor_bekas.csv'
data = pd.read_csv(file_path)

# Memilih kolom numerik
data_numerik = data.select_dtypes(include=['float64', 'int64'])

# Menampilkan informasi dataset
data.info()
print('Missing values per column:')
print(data.isnull().sum())

# Visualisasi nilai yang hilang
plt.figure(figsize=(8, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Dataset')
plt.show()

# Mengisi nilai yang hilang dengan median untuk kolom numerik
numerical_columns = data.select_dtypes(include=['number']).columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

# Encoding variabel kategorikal
data = pd.get_dummies(data, drop_first=True)

# Memisahkan fitur dan target
try:
    X = data.drop('harga', axis=1)
    y = data['harga']
    print('Shape fitur:', X.shape)
    print('Shape target:', y.shape)
except KeyError as e:
    print(f"Error: {e}")
    print("Kolom yang tersedia dalam dataset:", data.columns)

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Menyimpan model ke file model.pkl
dump(model, 'model.pkl')
print('Model telah disimpan ke file model.pkl')

# Memuat model dari file model.pkl
loaded_model = load('model.pkl')
print('Model telah dimuat dari file model.pkl')

# Prediksi pada data test
y_pred = loaded_model.predict(X_test)

# Evaluasi model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared Score: {r2}')

# Visualisasi hubungan prediksi dan nilai aktual
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='purple')
plt.title('Hubungan Nilai Aktual dan Prediksi')
plt.xlabel('Nilai Aktual')
plt.ylabel('Nilai Prediksi')
plt.show()

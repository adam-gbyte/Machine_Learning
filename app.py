import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    data = pd.read_csv("passwords.csv")
    return data

data = load_data()

def extract_features(password):
    length = len(password)
    upper = sum(1 for c in password if c.isupper())
    digits = sum(1 for c in password if c.isdigit())
    symbols = sum(1 for c in password if not c.isalnum())
    return [length, upper, digits, symbols]

X = data['password'].apply(extract_features).tolist()
y = data['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

st.title("Klasifikasi Tingkat Keamanan Password")
st.write("Menggunakan Algoritma KNN berdasarkan Pola Penulisan dan Kompleksitas")

user_password = st.text_input("Masukkan Password Anda", type="password")

def calculate_score(password):
    length = len(password)
    upper = sum(1 for c in password if c.isupper())
    digits = sum(1 for c in password if c.isdigit())
    symbols = sum(1 for c in password if not c.isalnum())

    score = 0
    score += min(length * 5, 30)
    score += min(upper * 5, 20)
    score += min(digits * 5, 20)
    score += min(symbols * 7, 30)

    return min(score, 100)

deskripsi_strength = {
    0: '🔴 Password sangat lemah, mudah ditebak atau dibobol. Sebaiknya gunakan kombinasi huruf besar, angka, dan simbol.',
    1: '🟡 Password cukup aman, tapi masih bisa ditingkatkan keamanannya.',
    2: '🟢 Password sangat kuat dan aman dari serangan umum.'
}

if user_password:
    features = np.array(extract_features(user_password)).reshape(1, -1)
    prediction = knn.predict(features)[0]
    score = calculate_score(user_password)

    st.subheader("Hasil Klasifikasi:")
    st.write(f"**Tingkat Keamanan Password: {prediction}**")
    st.write(deskripsi_strength[prediction])
    st.progress(score)
    st.write(f"Score: **{score}/100**")

    st.subheader("Detail Fitur Password:")
    st.write(f"Panjang: {len(user_password)}")
    st.write(f"Jumlah Huruf Kapital: {sum(1 for c in user_password if c.isupper())}")
    st.write(f"Jumlah Angka: {sum(1 for c in user_password if c.isdigit())}")
    st.write(f"Jumlah Simbol: {sum(1 for c in user_password if not c.isalnum())}")
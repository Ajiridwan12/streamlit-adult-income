import streamlit as st

def app():
    st.markdown(
        """
        <style>
        .css-2trqyj {
            font-family: 'Times New Roman', Times, serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("- Aplikasi Prediksi Adult Income -")
    st.subheader("Selamat Datang Di Aplikasi Prediksi Adult income!")
    st.write("Aplikasi prediksi adult income adalah aplikasi yang digunakan untuk memprediksi proses evaluasi dan pemahaman terhadap sumber dan jumlah pendapatan yang diterima oleh individu, keluarga, perusahaan, atau pemerintah dalam suatu periode waktu tertentu.")
    st.image("2.jpg")
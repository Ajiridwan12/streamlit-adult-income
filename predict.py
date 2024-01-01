import numpy as np
from web_functions import predict
from web_functions import load_data
import streamlit as st
import pickle



def app(df, X, y):
    st.title("Input Nilai Untuk Prediction Dengan Algoritma KNN")
    aid_model = pickle.load(open('AID.sav','rb'))

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=150, value=0)
    with col1:
        workclass = st.number_input('Input workclass')
    with col1:
        education = st.number_input('education')
    with col1:
        marital_status = st.number_input('marital')
    with col2:
        occupation = st.number_input('occupation')
    with col2:  
        relationship = st.number_input('realtionship')
    with col2:
        gender = st.number_input('gender')
    with col2:
        hours = st.number_input('hours per week')
    with col2:
        native_country = st.number_input('native')
        

    ai_pred = ''
    if st.button('Klasifikasi Income:') :
        aid_prediction = aid_model.predict([[age,workclass,education,marital_status,occupation,relationship,gender,hours,native_country]])

        if(aid_prediction[0] == 0):
            aid_pred = 'Income <=50k'
        else :
            aid_pred = 'Income =>50k'

        st.success(aid_pred)

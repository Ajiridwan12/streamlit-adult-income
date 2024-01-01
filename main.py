import streamlit as st
from web_functions import load_data
from home import app as home 
from predict import app as predict
from visualise import app as visualise

Tabs = {
    "Home": home, 
    "Prediction": predict,  
    "Visualisation": visualise 
}

# membuat sidebar
st.sidebar.title("Navigasi")

# Membuat Radio Option
page = st.sidebar.radio("Pages", list(Tabs.keys()))

# Load dataset
df, X, y = load_data()  

# Kondisi untuk Memanggil App Function
if page in ["Prediction", "Visualisation"]:
    Tabs[page](df, X, y)  
else:
    Tabs[page]()  

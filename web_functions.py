import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler

@st.cache_data
def load_data():
    # load dataset
    df = pd.read_csv('adult.csv')

    label_encoder = preprocessing.LabelEncoder()

    df['gender'] = label_encoder.fit_transform(df['gender'])
    df['workclass'] = label_encoder.fit_transform(df['workclass'])
    df['education'] = label_encoder.fit_transform(df['education'])
    df['marital-status'] = label_encoder.fit_transform(df['marital-status'])
    df['occupation'] = label_encoder.fit_transform(df['occupation'])
    df['relationship'] = label_encoder.fit_transform(df['relationship'])
    df['native-country'] = label_encoder.fit_transform(df['native-country'])
    df['income'] = label_encoder.fit_transform(df['income'])

    X = df[['age', 'workclass', 'education', 'marital-status', 'occupation','relationship', 'gender', 'hours-per-week', 'native-country']]
    y = df["income"]  

    return df, X, y 

@st.cache_data
def train_model(X, y):
    model = KNeighborsClassifier()
    model.fit(X, y)

    score = model.score(X, y)

    return model, score  

def predict(X, y, features):
    model, score = train_model(X, y)

    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, score
  

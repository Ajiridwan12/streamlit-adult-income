import warnings
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from sklearn.decomposition import PCA 
from sklearn.metrics import confusion_matrix
import itertools 

from web_functions import train_model  

def plot_confusion_matrix(model, X, y):
    cm = confusion_matrix(y, model.predict(X))
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['Class 0', 'Class 1']  # Ganti dengan label kelas Anda jika diperlukan
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def app(df, X, y):
    warnings.filterwarnings('ignore') 
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("- Visualisasi -")  
    model, score = train_model(X, y)
    if st.checkbox("Plot Confusion Matrix"):
        plot_confusion_matrix(model, X, y)
        st.pyplot()
    if st.checkbox("Plot K-Neighboors"):  
        pca = PCA(n_components=2)
        X_test_pca = pca.fit_transform(X)  
        y_pred_test = model.predict(X)
        pca_df = pd.DataFrame({'income': X_test_pca[:, 0], 'age': X_test_pca[:, 1], 'Stroke': y_pred_test})
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_df['income'], pca_df['age'], c=pca_df['Stroke'], cmap='coolwarm')
        plt.title('PCA Visualization of KNN Predictions')
        plt.xlabel('Income')
        plt.ylabel('Age')
        st.pyplot()

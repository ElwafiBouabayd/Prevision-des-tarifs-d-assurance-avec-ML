import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Charger le modèle pré-entraîné
model = joblib.load("insurance_model.pkl")
scaler = joblib.load("scaler.pkl")  # Si nécessaire pour la mise à l'échelle des données

st.title("Prédiction du Prix de l'Assurance")

# Création des entrées utilisateur
age = st.number_input("Âge", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sexe", ["male", "female"])
bmi = st.number_input("IMC (Indice de Masse Corporelle)", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Nombre d'enfants", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Fumeur", ["yes", "no"])
region = st.selectbox("Région", ["southwest", "southeast", "northwest", "northeast"])

# Transformer les entrées utilisateur en dataframe
data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# Encodage des variables catégorielles
data = pd.get_dummies(data, drop_first=True)

# Mise à l'échelle si nécessaire
data_scaled = scaler.transform(data) if scaler else data

# Prédiction
if st.button("Prédire"):
    prediction = model.predict(data_scaled)
    st.success(f"Coût estimé de l'assurance : {prediction[0]:,.2f} USD")

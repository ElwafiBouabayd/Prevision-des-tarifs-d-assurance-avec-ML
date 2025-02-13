import streamlit as st
import pandas as pd
import joblib
from category_encoders import OneHotEncoder

model = joblib.load("modelXGBoost.pkl")
st.title("Prédiction des tarifs de l'assurance")


age = st.number_input("Âge", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sexe", ["male", "female"])
bmi = st.number_input("IMC (Indice de Masse Corporelle)", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Nombre d'enfants", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Fumeur", ["yes", "no"])
region = st.selectbox("Région", ["southwest", "southeast", "northwest", "northeast"])

data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

ohe = OneHotEncoder(use_cat_names=True)
data_ = ohe.fit_transform(data)
print(data_)

if st.button("Prédire"):
    prediction = model.predict(data_)
    st.success(f"Coût estimé de l'assurance : {prediction[0]:,.2f} USD")

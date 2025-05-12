import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("model.pkl")



# Styles personnalisés (ajout de CSS dans Streamlit)
st.markdown(
    """
    <style>
        body {
            background-color: #F4F6F9;  /* Fond clair et moderne */
            color: #2D3B44;  /* Couleur sombre pour un contraste agréable */
        }
        .stButton>button {
            background-color: #003B5C;  /* Bleu foncé de la banque */
            color: white;
            font-weight: bold;
            border-radius: 8px;
            height: 50px;
            width: 100%;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1D2D44;  /* Bleu plus foncé au survol */
        }
        .stTitle {
            color: #005B79;  /* Bleu vif pour les titres */
            font-weight: bold;
        }
        .stText {
            color: #3E4C59;  /* Texte gris foncé pour la lisibilité */
        }
        .stSidebar {
            background-color: #FFFFFF;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .stSidebar>div {
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True
)


st.title(" Prédiction d'achat de produit")

gender = st.selectbox("Genre", ["Male", "Female"])
age = st.slider("Âge", 18, 60, 25)
salary = st.number_input("Salaire estimé", min_value=1000, max_value=150000, value=40000)

if st.button("Prédire"):
    gender_val = 1 if gender == "Male" else 0
    input_data = np.array([[gender_val, age, salary]])
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.success(f"✅ L'utilisateur VA acheter (Probabilité : {proba:.2f}%)")
    else:
        st.error(f"❌ L'utilisateur NE va PAS acheter (Probabilité : {proba:.2f}%)")

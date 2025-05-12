import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# === CHARGEMENT DES FICHIERS ===

# Charger le modèle
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Fichier 'best_model.pkl' introuvable.")
    st.stop()

# Charger le scaler
try:
    with open('scaler_dt.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Fichier 'scaler_dt.pkl' introuvable.")
    st.stop()

# Charger les noms de colonnes utilisés à l'entraînement
try:
    with open('features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except FileNotFoundError:
    st.error("Fichier 'features.pkl' introuvable.")
    st.stop()

# === INTERFACE UTILISATEUR ===

st.title("🚗 Prédiction du Prix des Véhicules")

st.write("""
Cette application prédit le **prix estimé** d'un véhicule en fonction de ses caractéristiques.
""")

# Champs de saisie utilisateur
symboling = st.number_input("Symboling", min_value=-3, max_value=3, value=0)
fueltype = st.selectbox("Fuel Type", ['gas', 'diesel'])
aspiration = st.selectbox("Aspiration", ['std', 'turbo'])
doornumber = st.selectbox("Number of Doors", ['two', 'four'])
carbody = st.selectbox("Car Body", ['convertible', 'hatchback', 'sedan', 'wagon'])
drivewheel = st.selectbox("Drive Wheel", ['rwd', 'fwd', '4wd'])
enginelocation = st.selectbox("Engine Location", ['front', 'rear'])
wheelbase = st.number_input("Wheelbase", min_value=80.0, max_value=120.0, value=100.0)
enginesize = st.number_input("Engine Size", min_value=50.0, max_value=300.0, value=100.0)
fuelsystem = st.selectbox("Fuel System", ['mpfi', '2bbl', '1bbl', 'spdi', 'mfi'])
boreratio = st.number_input("Bore Ratio", min_value=2.0, max_value=5.0, value=3.0)
stroke = st.number_input("Stroke", min_value=2.0, max_value=5.0, value=3.0)
compressionratio = st.number_input("Compression Ratio", min_value=5.0, max_value=15.0, value=10.0)
horsepower = st.number_input("Horsepower", min_value=50, max_value=300, value=100)
peakrpm = st.number_input("Peak RPM", min_value=4000, max_value=7000, value=5000)
citympg = st.number_input("City MPG", min_value=10, max_value=50, value=20)
highwaympg = st.number_input("Highway MPG", min_value=10, max_value=50, value=25)

# === BOUTON DE PRÉDICTION ===

if st.button("🔍 Prédire le prix"):
    # Préparation des données
    user_input = pd.DataFrame([[symboling, fueltype, aspiration, doornumber, carbody,
                                drivewheel, enginelocation, wheelbase, enginesize,
                                fuelsystem, boreratio, stroke, compressionratio,
                                horsepower, peakrpm, citympg, highwaympg]],
                              columns=['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody',
                                       'drivewheel', 'enginelocation', 'wheelbase', 'enginesize',
                                       'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
                                       'horsepower', 'peakrpm', 'citympg', 'highwaympg'])

    user_input = pd.get_dummies(user_input)
    user_input = user_input.reindex(columns=feature_names, fill_value=0)
    user_input_scaled = scaler.transform(user_input)

    predicted_price = model.predict(user_input_scaled)

    # Résultat
    st.success(f"💰 Le prix estimé de ce véhicule est : **{predicted_price[0]:,.2f} USD**")

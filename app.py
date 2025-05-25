import streamlit as st
import pickle
import pandas as pd

# Load model and symptoms
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("symptoms.pkl", "rb") as f:
    symptoms = pickle.load(f)

def encode_symptoms(input_symptoms, all_symptoms):
    df = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for sym in input_symptoms:
        sym = sym.strip()
        if sym in df.columns:
            df.at[0, sym] = 1
    return df

# Streamlit UI
st.title("Doctor Specialist Recommender")
st.markdown("Enter 2 or 3 symptoms and get the most likely disease prediction.")

selected_symptoms = st.multiselect("Select your symptoms:", options=symptoms)

if st.button("Predict"):
    if len(selected_symptoms) < 1:
        st.warning("Please select at least one symptom.")
    else:
        input_vector = encode_symptoms(selected_symptoms, symptoms)
        prediction = model.predict(input_vector)[0]
        st.success(f"Predicted Disease: **{prediction}**")

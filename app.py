import streamlit as st
import pickle
import pandas as pd
import numpy as np

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
        proba = model.predict_proba(input_vector)[0]
        # top_idx = np.argmax(proba)
        # predicted_disease = model.classes_[top_idx]
        # confidence = round(proba[top_idx] * 100, 2)

        
        # sym_str = ", ".join(selected_symptoms[:-1]) + f", and {selected_symptoms[-1]}" if len(selected_symptoms) > 1 else selected_symptoms[0]
        # st.success(f"Your symptoms {sym_str} suggest that you have **{predicted_disease}** with **{confidence}%** confidence.")
        
        sorted_indices = np.argsort(proba)[::-1]  # descending order
        top_classes = model.classes_[sorted_indices[:3]]
        top_probs = proba[sorted_indices[:3]]

        # Remove "_" for better readibility
        selected_symptoms = [sym.replace("_"," ") for sym in selected_symptoms]
        # Format symptom list
        sym_str = ", ".join(selected_symptoms[:-1]) + f", and {selected_symptoms[-1]}" if len(selected_symptoms) > 1 else selected_symptoms[0]

        # Main prediction
        main_pred = top_classes[0]
        main_conf = round(top_probs[0] * 100, 2)
        st.success(f"Your symptoms {sym_str} suggest that you have **{main_pred}** with **{main_conf}%** confidence.")

        # Additional likely predictions
        st.markdown("Some other likely diseases are:")
        for i in range(1, 3):
            st.markdown(f"**{i}) {top_classes[i]}** ({round(top_probs[i]*100, 2)}% confidence)")
        
        
        diseaseSpecialist = pd.read_csv('doctor-disease.csv')
        diseaseSpecialist["Disease"] = diseaseSpecialist["Disease"].str.strip().str.lower()
        predicted_disease = main_pred.strip().lower()

        # Match disease and get the corresponding specialist
        specialist_row = diseaseSpecialist[diseaseSpecialist["Disease"] == predicted_disease]

        if not specialist_row.empty:
            specialist = specialist_row["Specialist"].values[0]
            st.info(f"Recommended specialist for **{main_pred}**: **{specialist}**")
        else:
            st.warning(f"No specialist information found for **{main_pred}**.")
        st.markdown("---")
        # st.markdown(f"**Do you want a more detailed explanation ?**")
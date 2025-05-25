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

def printPrompt(symptoms, top_classes, top_probs, specialist):
    prompt = (
        f"A user used a ML system that predicts diseases based on symptoms and recommends a specialist.\n"
        f"The user expressed the following symptoms: {', '.join(symptoms)}.\n"
        f"The system gave the following diseases:\n"
        f"1. {top_classes[0]} ({round(top_probs[0]*100,2)}%)\n"
        f"2. {top_classes[1]} ({round(top_probs[1]*100,2)}%)\n"
        f"3. {top_classes[2]} ({round(top_probs[2]*100,2)}%)\n"
        f"Based on high confidence, the system recommended seeing a **{specialist}**.\n"
        f"Please explain the system's reasoning in a simple way."
    )
    st.markdown("### 📋 Explanation Prompt")
    st.code(prompt)

# --- UI ---
st.title("Doctor Specialist Recommender")
st.markdown("Enter 2 or 3 symptoms and get the most likely disease prediction.")

selected_symptoms = st.multiselect("Select your symptoms:", options=symptoms)

if st.button("Predict"):
    if len(selected_symptoms) < 1:
        st.warning("Please select at least one symptom.")
    else:
        input_vector = encode_symptoms(selected_symptoms, symptoms)
        proba = model.predict_proba(input_vector)[0]
        sorted_indices = np.argsort(proba)[::-1]
        top_classes = model.classes_[sorted_indices[:3]]
        top_probs = proba[sorted_indices[:3]]

        selected_symptoms_clean = [sym.replace("_", " ") for sym in selected_symptoms]
        sym_str = ", ".join(selected_symptoms_clean[:-1]) + f", and {selected_symptoms_clean[-1]}" if len(selected_symptoms_clean) > 1 else selected_symptoms_clean[0]

        main_pred = top_classes[0]
        main_conf = round(top_probs[0] * 100, 2)
        
        # Get specialist
        diseaseSpecialist = pd.read_csv('doctor-disease.csv')
        diseaseSpecialist["Disease"] = diseaseSpecialist["Disease"].str.strip().str.lower()
        predicted_disease = main_pred.strip().lower()
        specialist_row = diseaseSpecialist[diseaseSpecialist["Disease"] == predicted_disease]

        if not specialist_row.empty:
            specialist = specialist_row["Specialist"].values[0]
            # st.info(f"Recommended specialist for **{main_pred}**: **{specialist}**")
        else:
            specialist = "Unknown"
            st.warning(f"No specialist information found for **{main_pred}**.")

        # Save to session state
        st.session_state.prediction_ready = True
        st.session_state.selected_symptoms_clean = selected_symptoms_clean
        st.session_state.top_classes = top_classes
        st.session_state.top_probs = top_probs
        st.session_state.specialist = specialist

        # Reset explain_clicked on new prediction
        st.session_state.explain_clicked = False

# Show prompt if "Yes, explain" is clicked
if st.session_state.get("prediction_ready", False):
    selected_symptoms_clean = st.session_state.selected_symptoms_clean
    top_classes = st.session_state.top_classes
    top_probs = st.session_state.top_probs
    specialist = st.session_state.specialist

    sym_str = ", ".join(selected_symptoms_clean[:-1]) + f", and {selected_symptoms_clean[-1]}" if len(selected_symptoms_clean) > 1 else selected_symptoms_clean[0]
    main_pred = top_classes[0]
    main_conf = round(top_probs[0] * 100, 2)

    st.success(f"Your symptoms {sym_str} suggest that you have **{main_pred}** with **{main_conf}%** confidence.")

    st.markdown("Some other likely diseases are:")
    for i in range(1, 3):
        st.markdown(f"**{i}) {top_classes[i]}** ({round(top_probs[i]*100, 2)}% confidence)")

    st.info(f"Recommended specialist for **{main_pred}**: **{specialist}**")

    st.markdown("---")
    st.markdown("**Do you want a more detailed explanation?**")
    if st.button("Yes, explain"):
        st.session_state.explain_clicked = True

if st.session_state.get("explain_clicked", False):
    printPrompt(
        st.session_state.selected_symptoms_clean,
        st.session_state.top_classes,
        st.session_state.top_probs,
        st.session_state.specialist
    )
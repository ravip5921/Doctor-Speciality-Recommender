import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import json

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

def make_prompt(symptoms, top_classes, top_probs, specialist):
   return f"""
            Objective: Provide explanations of the reasoning behind doctor specialist recommendations based on a simple model that predicts disease based on disease symptoms and assigns a specialist. 
            The goal is to enhance users' comprehension of how their symptoms can be related with those of some disease and why they should consult a particular specialist. 

            Input:
            - User reported symptoms: {', '.join(symptoms)}
            - Top 3 predicted diseases: 
            1. {top_classes[0]} ({round(top_probs[0]*100, 2)}%)
            2. {top_classes[1]} ({round(top_probs[1]*100, 2)}%)
            3. {top_classes[2]} ({round(top_probs[2]*100, 2)}%)
            - Recommended specialist: {specialist} 

            Expected Outcome:
            Users should gain a clear understanding of why specific diseases were predicted, with explanations tailored to their symptom profile. Users should be able to grasp the concepts of probability-based recommendation and symptom-disease matching.

            Guidelines:
            - Do not provide overly technical jargon unless asked by the user.
            - Do not give lengthy explanations; keep responses short, concise, and user-friendly.
            - Do not assume the user understands complex medical concepts; provide examples when necessary.
        """.strip()

def send_prompt_to_llm(prompt):
    # call LLM endpoint
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    try:
        with requests.post(API_URL, json=payload, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            llm_reply = ""
            container = st.empty()
            for line in resp.iter_lines():
                if line:
                    try:
                        obj = json.loads(line.decode('utf-8'))
                        content_piece = obj.get("message", {}).get("content", "")
                        llm_reply += content_piece
                        container.markdown(f"### Explanation for Recommendation\n\n{llm_reply}")
                    except json.JSONDecodeError:
                        pass
            return llm_reply
    except Exception as e:
        llm_reply = f"API request failed: {e}"
        return None

# --- API ---
API_URL = "http://m2025.cht77.com:1334/api/chat"
MODEL_NAME = "llama3.3:70b-instruct-q8_0"

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
        # build prompt
        prompt = make_prompt(
            st.session_state.selected_symptoms_clean,
            st.session_state.top_classes,
            st.session_state.top_probs,
            st.session_state.specialist
        )
        send_prompt_to_llm(prompt)
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import json
import streamlit.components.v1 as components

API_URL = st.secrets["api"]["url"]
MODEL_NAME = st.secrets["api"]["model"]

# Load model and symptoms
with open("model/disease-model.pkl", "rb") as f:
    disease_model = pickle.load(f)

with open("model/symptoms.pkl", "rb") as f:
    symptoms = pickle.load(f)

with open("model/specialist-model.pkl", "rb") as f:
    specialist_model = pickle.load(f)
with open("model/mlb.pkl","rb") as f:
    mlb = pickle.load(f)


# --- Session state inits ---
for key, default in [
    ("prediction_ready", False),
    ("initial_prompt_sent", False),
    ("chat_history", []),
    ("chat_html", ""),
    ("explain_clicked", False),
    ("selected_symptoms_clean", None),
    ("show_explain_option",False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

def encode_symptoms(input_symptoms, all_symptoms):
    df = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for sym in input_symptoms:
        sym = sym.strip()
        if sym in df.columns:
            df.at[0, sym] = 1
    return df

def make_prompt(symptoms, top_classes, top_probs, specialist, specialist_prob):
   return f"""
            Provide explanations of the reasoning behind doctor specialist recommendations based on a system that comprises two simple models, the first one predicts diseases based on disease symptoms and outputs top three diseases, the second model takes the three disease labels as input and assigns a specialist. 
            The goal is to enhance users' comprehension of how their symptoms can be related with those of some disease and why they should consult a particular specialist. 
            Below is the **exact system design** as implemented in our code:
            
            1. Disease Prediction Model (Random Forest):
            - Input features: a binary symptom vector indicating presence (1) or absence (0) of each reported symptom.
            - Architecture: RandomForestClassifier with 200 decision trees (n_estimators=200, random_state=42).
            - For each tree:
                • It uses Gini impurity to choose splits based on symptom presence.
                • Each leaf node assigns a probability distribution over the set of possible diseases (proportion of training samples of each class).
            - Final output: For an input symptom vector, each tree yields a class-probability vector; the Random Forest's disease probability is the average over all trees.

            2. Specialist Recommendation Model (Multinomial Logistic Regression):
            - Input features: one-hot encoding of the top 3 predicted disease labels from the Random Forest (i.e., a vector of length equal to the number of diseases, with ones at each of the three disease indices).
            - Architecture: LogisticRegression(multi_class="multinomial", max_iter=500, class_weight="balanced").
            - Model mechanics:
                • Learns a weight matrix W and bias vector b such that for each specialist j, the model computes z_j = W_j·x + b_j.
                • Applies the softmax function over z_j to obtain specialist probabilities.
            - Final output: a probability distribution over specialists, from which the top 2 are selected.

            System Inputs and Outputs for This Session:

            Input:
            - User reported symptoms: {', '.join(symptoms)}
            - Top 3 predicted diseases: 
            1. {top_classes[0]} ({round(top_probs[0]*100, 2)}%)
            2. {top_classes[1]} ({round(top_probs[1]*100, 2)}%)
            3. {top_classes[2]} ({round(top_probs[2]*100, 2)}%)
            - Recommended specialist: 
            1. {specialist[0]} ({round(specialist_prob[0]*100, 2)}%)
            2. {specialist[1]} ({round(specialist_prob[0]*100, 2)}%)

            Expected Outcome:
            Users should gain a clear understanding of why specific diseases were predicted, with explanations tailored to their symptom profile. Users should be able to grasp the concepts of probability-based recommendation and symptom-disease matching.

            Guidelines:
            - Do not provide overly technical jargon unless asked by the user.
            - Do not give lengthy explanations; keep responses short, concise, and user-friendly.
            - Do not assume the user understands complex medical concepts; provide examples when necessary.
        """.strip()

def stream_to_llm(history, container):
    payload = {"model": MODEL_NAME, "messages": history}
    try:
        with requests.post(API_URL, json=payload, stream=True, timeout=60) as r:
            r.raise_for_status()
            current = "<div class='assistant-msg'><b>🤖</b> "
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                piece = obj.get("message", {}).get("content", "")
                current += piece
                html_code  = f"""<div id="chat" class='scrollbox'>{st.session_state.chat_html + current}</div>
                                <script>
                                    var chat = document.getElementById("chat");
                                    chat.scrollTop = chat.scrollHeight;
                                    console.log("hi");
                                </script>"""
                # update container
                container.markdown(html_code,
                                   unsafe_allow_html=True)
            # close this assistant block
            current += "</div>"
            # Strip HTML before storing in chat history
            rawChat = current.replace("<div class='assistant-msg'><b>🤖</b> ", "").replace("</div>", "")
            st.session_state.chat_html += current
            # Append raw text to history
            st.session_state.chat_history.append({"role": "assistant", "content": rawChat})
            return True
    except Exception as e:
        err = f"<div class='assistant-msg'><b>🤖</b> Error: {e}</div>"
        st.session_state.chat_html += err
        html_code = f"""<div id="chat" class='scrollbox'>{st.session_state.chat_html}</div>
                        <script>
                           var chat = document.getElementById("chat");
                            chat.scrollTop = chat.scrollHeight;
                            console.log("hi");
                        </script>"""
        container.markdown(html_code,unsafe_allow_html=True)
        return False

# --- UI ---
st.title("Doctor Specialist Recommender")
st.markdown("Select your symptoms and get the three most likely diseases prediction.")

selected_symptoms = st.multiselect("Select your symptoms:", options=symptoms)

if st.button("Predict"):
    if len(selected_symptoms) < 1:
        st.warning("Please select at least one symptom.")
    else:
        input_vector = encode_symptoms(selected_symptoms, symptoms)
        proba = disease_model.predict_proba(input_vector)[0]
        sorted_indices = np.argsort(proba)[::-1]
        top_classes = disease_model.classes_[sorted_indices[:3]]
        top_probs = proba[sorted_indices[:3]]

        selected_symptoms_clean = [sym.replace("_", " ") for sym in selected_symptoms]
        sym_str = ", ".join(selected_symptoms_clean[:-1]) + f", and {selected_symptoms_clean[-1]}" if len(selected_symptoms_clean) > 1 else selected_symptoms_clean[0]

        top3_diseases = list(top_classes)

        # Only use top 3 labels for specialist prediction
        X_spec_input = mlb.transform([top3_diseases])
        
        specialist = specialist_model.predict_proba(X_spec_input)[0]
        sorted_indices_sp = np.argsort(specialist)[::-1]
        top_classes_sp = specialist_model.classes_[sorted_indices_sp[:3]]
        top_probs_sp = specialist[sorted_indices_sp[:3]]

        # Save to session state
        st.session_state.prediction_ready = True
        st.session_state.selected_symptoms_clean = selected_symptoms_clean
        st.session_state.top_classes = top_classes
        st.session_state.top_probs = top_probs
        st.session_state.specialists = top_classes_sp
        st.session_state.specialists_pb = top_probs_sp
        st.session_state.show_explain_option = True

        st.session_state.initial_prompt_sent = False
        st.session_state.chat_history = []
        st.session_state.chat_html = ""
        st.session_state.explain_clicked = False


# --- Show Prediction ---
if st.session_state.prediction_ready:
    selected_symptoms_clean = st.session_state.selected_symptoms_clean
    top_classes = st.session_state.top_classes
    top_probs = st.session_state.top_probs
    specialist = st.session_state.specialists
    specialist_prob = st.session_state.specialists_pb
    specialist_prob = [round(x * 100, 2) for x in specialist_prob]

    sym_str = ", ".join(selected_symptoms_clean[:-1]) + f", and {selected_symptoms_clean[-1]}" if len(selected_symptoms_clean) > 1 else selected_symptoms_clean[0]
    main_pred = top_classes[0]
    main_conf = round(top_probs[0] * 100, 2)

    st.success(f"Your symptoms {sym_str} suggest that you might have the following diseases with respective confidence as shown:")

    st.markdown("Top likely diseases are:")
    for i in range(0, 3):
        st.markdown(f"**{i+1}) {top_classes[i]}** ({round(top_probs[i]*100, 2)}% confidence)")

    st.info(f"For theses diseases, our **Specialist Recommendation** model suggests: \n\n- **{specialist[0]}** ({specialist_prob[0]} % confidence)\n- **{specialist[1]}** ({specialist_prob[1]} % confidence)")

    st.markdown("---")

# if not st.session_state['explain_clicked']:
#     st.markdown(st.session_state['explain_clicked']) # -> False
#     if st.session_state.prediction_ready:
#         st.markdown("**Do you want a more detailed explanation?**")
#         if st.button("Yes, explain") and not st.session_state.initial_prompt_sent:
#             st.session_state.explain_clicked = True

explain_container = st.empty()
if st.session_state.show_explain_option:
    explain_container.markdown("**Do you want a more detailed explanation?**")
    if explain_container.button("Get Explanation"):
        st.session_state.show_explain_option = False
        st.session_state.explain_clicked = True
        explain_container.empty()
        
# --- Explanation and Follow ups ---
if st.session_state.explain_clicked:
    st.markdown("### 💬 Explanation and Follow-ups")



chat_box = st.empty()
if st.session_state.initial_prompt_sent or st.session_state.explain_clicked:
    st.markdown("""
    <style>
        .scrollbox {
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 10px;
            background-color: #f9f9f9;
            font-size: 0.95rem;
        }
        .user-msg { color: #0b0c0c; margin-bottom: 0.5em; background-color: gainsboro; padding-top: 10px; padding-bottom: 10px;}
        .assistant-msg { color: #009966; margin-bottom: 1em; }
    </style>
    """, unsafe_allow_html=True)
    chat_box.markdown("<div id='chat' class='scrollbox'></div>", unsafe_allow_html=True)

if st.session_state.explain_clicked:
    prompt = make_prompt(selected_symptoms_clean, top_classes, top_probs, specialist, specialist_prob)
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful medical explainer who explans the decision of two multiclass classifiers. The first one predicts user's top 3 likely diseases based on input symtoms and the second one recommends a specialist based on the top 3 predicted likely diseases."},
        {"role": "user",   "content": prompt}
    ]    
    st.session_state.initial_prompt_sent = True
    st.session_state.explain_clicked = False
    

    # Start LLM streaming explanation
    stream_to_llm(st.session_state.chat_history, chat_box)

# --- Follow-up form and stream response---
if st.session_state.initial_prompt_sent:
    chat_box.markdown(f"<div class='scrollbox'>{st.session_state.chat_html}</div>",
                      unsafe_allow_html=True)

    with st.form("followup"):
        cols = st.columns([ 4, 0.5]) 
        # cols[0].write("💬")
        followup = cols[0].text_input("", label_visibility="collapsed")
        send = cols[1].form_submit_button("➤")

    if send and followup:
        st.session_state.chat_history.append({"role":"user","content":followup})
        st.session_state.chat_html += f"<div class='user-msg'><b>🧑</b>{' ' + followup}</div>"
        # re-render with user msg
        chat_box.markdown(f"<div class='scrollbox'>{st.session_state.chat_html}</div>",
                          unsafe_allow_html=True)

        # stream the assistant reply
        stream_to_llm(st.session_state.chat_history, chat_box)
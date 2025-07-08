import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import json
import random

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

# Load Patient Scenarios

# Load scenarios from markdown file
def load_scenarios(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        scenarios = file.read().split("---")  # "---" as a separator
    scenarios = [ x.strip() for x in scenarios]
    return scenarios


# --- Session state inits ---
for key, default in [
    ("page","home"),
    ("user_name",""),
    ("prediction_ready", False),
    ("initial_prompt_sent", False),
    ("chat_history", []),
    ("chat_html", ""),
    ("explain_clicked", False),
    ("selected_symptoms_clean", None),
    ("show_explain_option",False),
    ("scenarios_loaded",False)
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --- Model outputs ---
for key, default in [
    ("top_classes", []),
    ("top_probs", []),
    ("specialists", []),
    ("specialists_pb", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def encode_symptoms(input_symptoms, all_symptoms):
    """
    Given a list of selected symptom names and a master list of all symptom columns,
    return a single-row DataFrame of zeros/ones indicating which symptoms are present.
    """
    df = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for sym in input_symptoms:
        sym = sym.strip()
        if sym in df.columns:
            df.at[0, sym] = 1
    return df

def make_prompt(symptoms, top_classes, top_probs, specialist, specialist_prob, scenario):
    """
    Construct the system prompt explaining the two-model 
    architecture and the specific inputs/outputs for this session.
    """
    return f"""
            Provide explanations of the reasoning behind doctor specialist recommendations based on a system that comprises two simple models, the first one predicts diseases based on disease symptoms and outputs top three diseases, the second model takes the three disease labels as input and assigns a specialist. 
            The users are presented with a patient scenario and they input symptoms based on that particular scenario to the system to get recommendations.
            The goal is to enhance users' comprehension of how the symptoms can be related with those of some disease and why the patient should consult a particular specialist. 
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
            - Patient Scenario: {scenario}
            - User reported symptoms: {', '.join(symptoms)}
            - Top 3 predicted diseases: 
            1. {top_classes[0]} ({round(top_probs[0]*100, 2)}%)
            2. {top_classes[1]} ({round(top_probs[1]*100, 2)}%)
            3. {top_classes[2]} ({round(top_probs[2]*100, 2)}%)
            - Recommended specialist: 
            1. {specialist[0]} ({round(specialist_prob[0]*100, 2)}%)
            2. {specialist[1]} ({round(specialist_prob[1]*100, 2)}%)

            Expected Outcome:
            Users should gain a clear, intuitive sense of how their specific symptoms drove the model's decisions, explore how tweaks to those symptoms would change the output, see a simple example of how a few decision trees vote, and understand what the confidence scores actually mean.

            Guidelines:
            - Do not provide overly technical jargon unless asked by the user.
            - Do not give lengthy explanations; keep responses short, concise, and user-friendly.
            - Do not assume the user understands complex medical concepts; provide examples when necessary.
            - Weave in **feature-based insight**: mention which symptoms had the strongest influence on each disease probability.
            - Provide **multiple reasonable counterfactual scenarios**: describe how altering one or two symptoms (adding or removing) could shift the rankings in different directions. Offer at least two plausible "what-if" examples illustrating how predictions might change.
            - Briefly mention that the system uses a Random Forest of many decision trees for disease prediction, then illustrate with a **simple example** of one or two decision trees "voting" step by step (e.g., "Tree #1 checks symptom1 → symptom2 → symptom3 and casts its vote for disease1. Also explain the specialist model in a similar way.
            - Clarify **how to read the output**: what the percentages represent and why close scores still merit attention.

            Now, using this system description and the inputs above, please generate an explanation that naturally covers all of the above elements.
        """.strip()

def stream_to_llm(history, container):
    """
    Send the `history` list of {"role": "...", "content": "..."} messages 
    to the LLM API and append the assistant's streamed response to `chat_html` 
    and `chat_history`.
    """
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

# HOME PAGE

if st.session_state.page == "home":
    st.title("Doctor Specialist Recommender")
    if st.session_state.user_name == "":
        st.markdown("### Please enter your name to get started:")

        # 4.1) Name input
        name_col, submit_col = st.columns([3, 1], vertical_alignment="bottom")
        with name_col:
            if st.session_state["user_name"] == "":
                st.session_state["user_name"] = st.text_input("Your name:", value="", placeholder="Type your name here")
        with submit_col:
            if st.button("Start"):
                if st.session_state["user_name"].strip() == "":
                    st.warning("Please enter at least one character for your name.")
                else:
                    st.success(f"Hello, {st.session_state['user_name'].strip()}!")
                    st.rerun()
    
    if not st.session_state.scenarios_loaded:            
        scenarios = load_scenarios("patient-scenarios.md")
        random.shuffle(scenarios)
        # Keep 3 scenarios
        st.session_state.selected_scenarios = scenarios[:3] if len(scenarios) >= 3 else scenarios
        st.session_state.scenarios_loaded = True
    # Only show the welcome text & cards if we have a name
    if st.session_state["user_name"].strip() != "":
        st.markdown(f"#### Hi, **{st.session_state['user_name']}**! Please choose a version below:")
        st.markdown("")

        # 4.2) Three card-style buttons
        c1, c2, c3 = st.columns(3, gap="large")

        with c1:
            st.info("""**Version 1**: Prediction Model.
                    \nGet diseases predictions & specialists recommendations.""")
            if st.button("Go to Version 1"):
                st.session_state.page = "v1"
                st.rerun()
                

        with c2:
            st.info("""**Version 2**: Prediction Model and AI Response.
                    \nEnter symptoms, get diseases & specialists recommendation, and ask specified follow up questions.""")
            if st.button("Go to Version 2"):
                st.session_state.page = "v2"
                st.session_state.page = "v2"
                # Clear any previous state for v2
                st.session_state.prediction_ready = False
                st.session_state.initial_prompt_sent = False
                st.session_state.chat_history = []
                st.session_state.chat_html = ""
                st.session_state.explain_clicked = False
                st.rerun()
                

        with c3:
            st.info("""**Version 3**: Prediction Model and AI Chatbot.
                    \nEnter symptoms, get diseases & specialists recommendation, and chat for detailed explanations.""")
            if st.button("Go to Version 3"):
                st.session_state.page = "v3"
                # Clear any previous state for v3
                st.session_state.prediction_ready = False
                st.session_state.initial_prompt_sent = False
                st.session_state.chat_history = []
                st.session_state.chat_html = ""
                st.session_state.explain_clicked = False
                st.rerun()
                

elif st.session_state.page == "v3":
    # --- UI ---
    back_col, _ = st.columns([1, 4])
    with back_col:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.session_state.prediction_ready = False
            st.session_state.initial_prompt_sent = False
            st.session_state.chat_history = []
            st.session_state.chat_html = ""
            st.session_state.explain_clicked = False
            st.session_state.show_explain_option = False
            st.rerun()
    
    st.title("Doctor Specialist Recommender")
    st.subheader("Version 3 - Prediction Model with AI Chatbot")
    st.divider()

    st.markdown("_Patient Scenario:_")
    scenario = st.session_state.selected_scenarios[2]
    st.info(scenario)
    st.markdown("Select relevant symptoms for this case and get system's recommendations.")

    selected_symptoms = st.multiselect("Select symptoms:", options=symptoms)

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
            # CHECK
            st.rerun()

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
            st.markdown(f"**{i+1}) {top_classes[i]}** ({round(top_probs[i]*100, 2)} % confidence)")

        st.info(f"For these diseases, our **Specialist Recommendation** model suggests: \n\n- **{specialist[0]}** ({specialist_prob[0]} % confidence)\n- **{specialist[1]}** ({specialist_prob[1]} % confidence)")

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

            # check
            st.rerun()
            
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
        prompt = make_prompt(selected_symptoms_clean, top_classes, top_probs, specialist, specialist_prob, scenario)
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

        with st.form("followup",clear_on_submit=True):
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


elif st.session_state.page == "v1":
    # --- UI ---
    back_col, _ = st.columns([1, 4])
    with back_col:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.session_state.prediction_ready = False
            st.session_state.initial_prompt_sent = False
            st.session_state.chat_history = []
            st.session_state.chat_html = ""
            st.session_state.explain_clicked = False
            st.session_state.show_explain_option = False
            st.rerun()
    
    st.title("Doctor Specialist Recommender")
    st.subheader("Version 1 - Prediction Model")
    st.divider()

    st.markdown("_Patient Scenario:_")
    scenario = st.session_state.selected_scenarios[0]
    st.info(scenario)
    st.markdown("Select relevant symptoms for this case and get system's recommendations.")

    selected_symptoms = st.multiselect("Select symptoms:", options=symptoms)

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
            # CHECK
            st.rerun()

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

        st.markdown("## 📋 How the System Works")

        st.markdown("""
        When you enter symptoms, our system first predicts which diseases are most likely, then recommends which specialists you should consult. 
        This happens in two steps: a **Random Forest** model with 200 decision trees predicts the likely diseases, and a **Logistic Regression** model takes those (top 3) diseases and ranks the top 2 specialists.
        
        The Random Forest works by having many simple decision trees, each making its own prediction based on your symptoms.
        The trees *vote*, and their predictions are averaged to produce more stable and accurate results than relying on a single tree.
        Symptoms that often co-occurred with particular diseases in the training data, the examples the models learned from, tend to carry more weight in the prediction. 
        However, some diseases may still appear with lower confidence scores because the system considers all plausible options, even less likely ones, rather than ignoring them completely. 
        
        The specialist recommendation is then made by the Logistic Regression model, which assigns probabilities to all specialists based on how well the top three predicted diseases fit their expertise.
        It uses a mathematical function called *softmax* to distribute probabilities across all specialists, ensuring even lower-ranked ones still receive some score, reflecting that they are less likely but still plausible.

        Together, these models aim to balance accuracy and coverage, providing recommendations that are reliable yet still account for less obvious possibilities.
        """)

        with st.expander("**Key Terms**", expanded=True):
            st.markdown("""
        - **Decision Tree:** A simple rule-based model that asks yes/no about each symptom and votes for a disease.
        - **Random Forest:** A group of 200 decision trees. Each votes, and their results are averaged for more reliable predictions.
        - **Logistic Regression:** A statistical model that scores each specialist based on the predicted diseases.
        - **Confidence Score:** The percentage shown next to each disease or specialist. The higher it is, the stronger the model's belief.
        - **Binary Vector:** A list of 1s and 0s *[Showing which symptoms you reported (1 if present, 0 if not)]*.
        """)

        # with st.expander("**Why the System Works This Way**"):
        #     st.markdown("""
        # - Using many trees reduces errors compared to one tree and makes predictions more stable.
        # - Even diseases or specialists with low scores appear because the model considers all plausible options.
        # - Some symptoms strongly point to certain diseases because they co-occurred in many training examples.
        # - Not all specialists are recommended if no disease strongly aligns with their area.
        # - Logistic Regression distributes probabilities across all specialists using a softmax function.
        # - Training data is a set of past examples the models learned from.
        # """)
        

        st.markdown("""
        After reading this, you can proceed to the quiz.
        """)

        st.markdown("---")

        # Link to Quiz
        st.markdown(
            """
            <a href="https://quiz-doctor-speciality-recommender.streamlit.app/" target="_blank">
                <button style="
                    background-color:#4CAF50;
                    border:none;
                    color:white;
                    padding:10px 20px;
                    text-align:center;
                    text-decoration:none;
                    display:inline-block;
                    font-size:16px;
                    border-radius:5px;
                    cursor:pointer;">
                    Go to Quiz
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )


elif st.session_state.page == "v2":
    # --- UI ---
    back_col, _ = st.columns([1, 4])
    with back_col:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.session_state.prediction_ready = False
            st.session_state.initial_prompt_sent = False
            st.session_state.chat_history = []
            st.session_state.chat_html = ""
            st.session_state.explain_clicked = False
            st.session_state.show_explain_option = False
            st.session_state.followup_idx = 0
            st.rerun()
    
    st.title("Doctor Specialist Recommender")
    st.subheader("Version 2 - Prediction Model with AI Chatbot")
    st.divider()

    st.markdown("_Patient Scenario:_")
    scenario = st.session_state.selected_scenarios[1]
    st.info(scenario)
    st.markdown("Select relevant symptoms for this case and get system's recommendations.")

    selected_symptoms = st.multiselect("Select symptoms:", options=symptoms)

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
            st.session_state.followup_idx = 0
            st.rerun()

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

    explain_container = st.empty()
    if st.session_state.show_explain_option:
        explain_container.markdown("**Do you want a more detailed explanation?**")
        if explain_container.button("Get Explanation"):
            st.session_state.show_explain_option = False
            st.session_state.explain_clicked = True
            explain_container.empty()

            # check
            st.rerun()
            
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
        prompt = make_prompt(selected_symptoms_clean, top_classes, top_probs, specialist, specialist_prob, scenario)
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful medical explainer who explans the decision of two multiclass classifiers. The first one predicts user's top 3 likely diseases based on input symtoms and the second one recommends a specialist based on the top 3 predicted likely diseases."},
            {"role": "user",   "content": prompt}
        ]    
        st.session_state.initial_prompt_sent = True
        st.session_state.explain_clicked = False
        

        # Start LLM streaming explanation
        stream_to_llm(st.session_state.chat_history, chat_box)
    # --- Guided Follow-Up Buttons (using on_click) ---
    if st.session_state.initial_prompt_sent:
        chat_box.markdown(
            f"<div class='scrollbox'>{st.session_state['chat_html']}</div>",
            unsafe_allow_html=True,
        )

        questions = [
            "Can you explain step by step how symptoms are mapped to disease predictions?",
            "How does the model decide the confidence percentages for each of the top three diseases?",
            "Once the diseases are predicted, how does the system recommend the top 2 specialists?",
            "What mathematical function does the system use to turn scores into probabilities for specialists?",
            "Can you give me a simple example with some symptoms and walk me through the entire prediction process?",
            "How would changing or removing one symptom affect the disease and specialist predictions?"
        ]
        idx = st.session_state.followup_idx

        def ask_and_advance(i):
            q = questions[i]
            # append user question
            st.session_state.chat_history.append({"role":"user","content":q})
            st.session_state.chat_html += f"<div class='user-msg'><b>🧑</b> {q}</div>"
            # re-render the full chat (so far)
            chat_box.markdown(
                f"<div class='scrollbox'>{st.session_state['chat_html']}</div>",
                unsafe_allow_html=True,
            )
            # stream LLM reply
            stream_to_llm(st.session_state.chat_history, chat_box)
            # move to next question
            st.session_state.followup_idx += 1

        if idx < len(questions):
            # give each button a unique key so Streamlit can track it
            st.button(
                questions[idx], 
                key=f"followup_btn_{idx}", 
                on_click=ask_and_advance, 
                args=(idx,)
            )
        else:
            st.markdown("All follow-up questions completed.")
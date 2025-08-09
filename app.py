import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import json
import random
from supabase import create_client

API_URL = st.secrets["api"]["url"]
MODEL_NAME = st.secrets["api"]["model"]

SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]


DEBUG = st.secrets["params"]["debug"]
def debug_log(msg):
    if DEBUG:
        st.write("DEBUG: ", msg)


COOLDOWN_TIME_LONG = 30
COOLDOWN_TIME_SHORT = 15
NO_COOLDOWN = 0

if DEBUG:
    COOLDOWN_TIME_LONG = 1
    COOLDOWN_TIME_SHORT = 1

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

# Create Supabase client for logs
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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

def make_system_prompt(symptoms, top_classes, top_probs, specialist, specialist_prob, scenario):
    """
    Construct the system prompt explaining the two-model 
    architecture and the specific inputs/outputs for this session.
    """
    return f"""
            You are a medical explainer/chatbot designed to provide explanations of the reasoning behind the results of a machine learning system explained below. 
            The users are presented with a patient scenario and they input symptoms based on that particular scenario to the system to get diseases predictions and specialists recommendations.
            The goal is to enhance users' comprehension of how the symptoms can be related with those of some disease and why the patient should consult the recommended specialists. 
            
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
            - You can answer both general questions about how the system works, and specific questions about this scenario.
            - When asked **general questions** (like: *"How does the system work?"*), explain the overall system and how the two models work:
                • A short introduction of the system design and then breifly define the technical terms
                • How symptoms → diseases (via Random Forest and trees voting and averaging).
                • How diseases → specialists (via Logistic Regression weights and softmax).
                • Give few short paragraphs at most, no scenario-specific details, no what-if examples.
                • Use simple metaphors if appropriate (e.g., trees voting, experts weighing in).
            - When asked **scenario-specific questions** (like: *"How did [patient] get these results?"*):
                • Link the reported symptoms to the top-3 predicted diseases.
                • Mention which symptoms most influenced each disease prediction.
                • Briefly describe how a few example trees voted (e.g., “Tree #1 checked X → Y → Z and voted for A”). Use concrete example trees and symptoms to illustrate the decision process, ensuring intuitive understanding.
                • Explain how the diseases led to the recommended specialists.
            - When asked **what-if scenarios** or asked for result explanations: (like: *"What if I had [different symptom]?"*, *Can you explain the results?*):
                • Provide at least two plausible “what-if” scenarios showing how changing symptoms might alter predictions.
                • Clarify what the percentages mean and why lower probabilities can still be important.
            - When explicitly asked, you may also provide full mathematical details.
            
            You are ready to answer any user question now.
        """.strip()

def stream_to_llm(history, container):
    """
    Send the `history` list of {"role": "...", "content": "..."} messages 
    to the LLM API and append the assistant's streamed response to `chat_html` 
    and `chat_history`.
    """
    payload = {"model": MODEL_NAME, "messages": history}
    assistant_plain = ""
    try:
        with requests.post(API_URL, json=payload, stream=True, timeout=60) as r:
            r.raise_for_status()
            current = "<div class='assistant-msg'><b>🤖</b> "
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                piece = obj.get("message", {}).get("content", "")
                if not piece:
                    continue

                # accumulate plain text
                assistant_plain += piece
                
                # update HTML
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
            return assistant_plain
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

def display_results(selected_symptoms_clean, top_classes, top_probs, specialists, specialists_pb, showStalemate=True):
    
    specialist_prob = [round(x * 100, 2) for x in specialists_pb]

    sym_str = ", ".join(selected_symptoms_clean[:-1]) + f", and {selected_symptoms_clean[-1]}" if len(selected_symptoms_clean) > 1 else selected_symptoms_clean[0]

    st.success(f"Your symptoms {sym_str} suggest that you might have the following diseases with respective confidence as shown:")

    st.markdown("Top likely diseases are:")
    for i in range(0, 3):
        st.markdown(f"**{i+1}) {top_classes[i]}** ({round(top_probs[i]*100, 2)}% confidence)")

    st.info(f"For theses diseases, our **Specialist Recommendation** model suggests: \n\n- **{specialists[0]}** ({specialist_prob[0]} % confidence)\n- **{specialists[1]}** ({specialist_prob[1]} % confidence)")

    if showStalemate:
        # Call second function for details
        display_stalemate_text()

def display_stalemate_text():
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

    st.markdown("""
    After reading this, you can proceed to the quiz.
    """)

    # Link to Quiz
    st.markdown("---")

    reveal_button_html = """
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
        """
    countdown_component_html("Please read the given text carefully", st.session_state.get("COOLDOWN_TIME_LONG", COOLDOWN_TIME_LONG), reveal_button_html)



# helper functions to log each message
def log_message(role, text):
    supabase.table("transcript_logs").insert({
        "scenario_log_id": st.session_state["scenario_log_id"],
        "role": role,
        "text": text
    }).execute()




# HOME PAGE
def render_home_page():
    """Main entry for home page logic."""
    st.title("Doctor Specialist Recommender")

    if st.session_state.user_name.strip() == "":
        render_name_input()

    load_scenarios_if_needed()

    if st.session_state.user_name.strip():
        ensure_scenario_log()
        render_version_cards()

def render_name_input():
    """Handles name input and validation."""
    st.markdown("### Please enter your name to get started:")

    name_col, submit_col = st.columns([3, 1], vertical_alignment="bottom")
    with name_col:
        st.session_state["user_name"] = st.text_input(
            "Your name:",
            value=st.session_state["user_name"],
            placeholder="Type your name here"
        )

    with submit_col:
        if st.button("Start"):
            debug_log("Start Pressed")
            user = st.session_state["user_name"].strip()
            if user == "":
                st.warning("Please enter at least one character for your name.")
                debug_log("User Name empty after start press")
            else:
                st.success(f"Hello, {st.session_state['user_name'].strip()}!")
            st.rerun()


def load_scenarios_if_needed():
    """Loads scenarios into session state if not already loaded."""
    if not st.session_state.scenarios_loaded:            
        scenarios = load_scenarios("patient-scenarios.md")
        random.shuffle(scenarios)
        st.session_state.selected_scenarios = scenarios[:3] if len(scenarios) >= 3 else scenarios
        st.session_state.scenarios_loaded = True
    # Set cooldown overrides for test user
    user = st.session_state["user_name"].strip()

    if user.lower() == "test":
        st.session_state["COOLDOWN_TIME_LONG"] = 5
        st.session_state["COOLDOWN_TIME_SHORT"] = 5
        st.session_state["NO_COOLDOWN"] = 5
    else:
        # Reset to defaults if needed
        st.session_state["COOLDOWN_TIME_LONG"] = COOLDOWN_TIME_LONG
        st.session_state["COOLDOWN_TIME_SHORT"] = COOLDOWN_TIME_SHORT
        st.session_state["NO_COOLDOWN"] = NO_COOLDOWN


def ensure_scenario_log():
    """Ensures a scenario log exists for the current user in DB."""
    if "scenario_log_id" not in st.session_state or not st.session_state["scenario_log_id"]:
        v2_scenario = st.session_state.selected_scenarios[1]
        v1_scenario = st.session_state.selected_scenarios[2]
        
        try:
            debug_log("Attempting Scenario insert")
            supabase.table("scenario_logs").insert({
                "username": st.session_state["user_name"].strip(),
                "scenario_v2": v2_scenario,
                "scenario_v1": v1_scenario
            }).execute()
            debug_log("Scenario Log inserted.")
        except Exception as e:
            debug_log(f"Insert failed: {e}")

        try:
            fetch_resp = (
                supabase.table("scenario_logs")
                .select("id")
                .eq("username", st.session_state["user_name"].strip())
                .order("started_at", desc=True)
                .limit(1)
                .execute()
            )
            st.session_state["scenario_log_id"] = fetch_resp.data[0]["id"]
            debug_log("Scenario ID stored.")
        except Exception as e:
            debug_log(f"Could not fetch scenario_log_id: {e}")

def reset_v1_state():
    """Resets all v1-specific states."""
    st.session_state.prediction_ready = False
    st.session_state.initial_prompt_sent = False
    st.session_state.chat_history = []
    st.session_state.chat_html = ""
    st.session_state.explain_clicked = False
    st.session_state.followup_idx = 1

def render_version_cards():
    """Renders version selection cards."""
    st.markdown(f"#### Hi, **{st.session_state['user_name']}**! Please choose a version below:")
    st.markdown("")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.info("""**Version 1**: Prediction Model and AI Response.
                \nGet diseases predictions & specialists recommendations, along with AI-generated explanations.\n\n You can ask the AI-chatbot some pre-selected and any questions you want.""")
        if st.button("Go to Version 1"):
            st.session_state.page = "v1"
            reset_v1_state()
            st.rerun()

    with c2:
        st.info("""**Version 2**: Prediction Model with PreQuiz.
                \nGet diseases predictions & specialists recommendations, along with AI-generated explanation and a pre quiz.\n\nThe pre-quiz will familiarize you with the system.""")
        if st.button("Go to Version 2"):
            st.session_state.page = "v2"
            st.rerun()

    

# COMMON UTILITIES
def reset_lock_timer():
    keys_to_delete = [
        key for key in st.session_state.keys() 
        if key.endswith("_done") or key.endswith("_end_time") or key == "unlock_time"
    ]
    for key in keys_to_delete:
        del st.session_state[key]

def render_back_button(page):
    back_col, _ = st.columns([1, 4])
    with back_col:
        if st.button("← Back to Home"):
            reset_common_state()
            reset_lock_timer()
            reset_prequiz_states()
            if page == "v1":
                st.session_state.followup_idx = 1
            st.session_state.page = "home"
            st.rerun()


def reset_common_state():
    st.session_state.prediction_ready = False
    st.session_state.initial_prompt_sent = False
    st.session_state.chat_history = []
    st.session_state.chat_html = ""
    st.session_state.explain_clicked = False
    st.session_state.show_explain_option = False


def reset_ai_state():
    st.session_state.initial_prompt_sent = False
    st.session_state.chat_history = []
    st.session_state.chat_html = ""
    st.session_state.explain_clicked = False

def reset_prequiz_states():
    keys_to_clear = [
        "v2_quiz_index",
        "v2_selected_options",
        "v2_quiz_done",
        "v2_chat_history_per_q",
        "v2_sent_system_prompt",
        "v2_initial_radio_set",
        "v2_input_used",
        "v2_quiz_questions"
    ]
    keys_to_clear += ["final_chat_history", "final_streaming", "v2_show_final_chat"]

    # Also remove any selected option keys per question
    for key in list(st.session_state.keys()):
        if key.startswith("selected_option_q_") or key.startswith("option_radio_q_") or key.startswith("form_q_") or key.startswith("input_q_"):
            keys_to_clear.append(key)

    for key in keys_to_clear:
        st.session_state.pop(key, None)


def render_scenario(version):
    st.markdown("_Patient Scenario:_")
    scenario = st.session_state.selected_scenarios[version]
    st.info(scenario)
    st.markdown("Select relevant symptoms for this case and get system's recommendations.")
    return scenario


def render_symptom_selector():
    return st.multiselect("Select symptoms:", options=symptoms)

def render_spacer():
    st.markdown("""
    <div style='min-height: 250px; overflow-y: auto; padding: 10px;'>
                <h1></h1>
    </div>
    """, unsafe_allow_html=True)

def handle_prediction(selected_symptoms):
    if len(selected_symptoms) < 1:
        st.warning("Please select at least one symptom.")
        return None

    selected_symptoms_clean = [sym.replace("_", " ") for sym in selected_symptoms]

    input_vector = encode_symptoms(selected_symptoms, symptoms)
    proba = disease_model.predict_proba(input_vector)[0]
    sorted_indices = np.argsort(proba)[::-1]
    top_classes = disease_model.classes_[sorted_indices[:3]]
    top_probs = proba[sorted_indices[:3]]

    X_spec_input = mlb.transform([list(top_classes)])
    specialist_proba = specialist_model.predict_proba(X_spec_input)[0]
    sorted_indices_sp = np.argsort(specialist_proba)[::-1]
    top_classes_sp = specialist_model.classes_[sorted_indices_sp[:3]]
    top_probs_sp = specialist_proba[sorted_indices_sp[:3]]

    st.session_state.prediction_ready = True
    st.session_state.selected_symptoms_clean = selected_symptoms_clean
    st.session_state.top_classes = top_classes
    st.session_state.top_probs = top_probs
    st.session_state.specialists = top_classes_sp
    st.session_state.specialists_pb = top_probs_sp
    st.session_state.show_explain_option = True
    reset_ai_state()
    return True

import time
from datetime import datetime, timedelta

def countdown_component_html(message, duration_sec, reveal_html):
    # Initialize unlock_time only when not already set
    if "unlock_time" not in st.session_state:
        st.session_state.unlock_time = datetime.now() + timedelta(seconds=duration_sec)

    remaining = int((st.session_state.unlock_time - datetime.now()).total_seconds())
    
    html_code = f"""
    <div style="font-weight:bold;font-size:16px;">
        <span id="timer">{message} — {remaining//60:02d}:{remaining%60:02d}</span>
    </div>

    <div id="reveal-section" style="display:none; margin-top:10px;">
        {reveal_html}
    </div>

    <script>
    var seconds = {remaining};
    var timerElement = document.getElementById("timer");
    var revealSection = document.getElementById("reveal-section");
    var countdown = setInterval(function(){{
        if (seconds > 0) {{
            seconds--;
            var mins = Math.floor(seconds/60);
            var secs = seconds % 60;
            timerElement.innerHTML = "{message} — " + 
                (mins<10?"0":"") + mins + ":" + (secs<10?"0":"") + secs;
        }} else {{
            clearInterval(countdown);
            timerElement.innerHTML = "You can now proceed!";
            revealSection.style.display = "block";
        }}
    }}, 1000);
    </script>
    """

    st.components.v2.html(html_code, height=120)
def countdown_with_button(message, duration_sec, button_label, button_key):
    # Initialize countdown state
    if f"{button_key}_done" not in st.session_state:
        st.session_state[f"{button_key}_done"] = False

    if not st.session_state[f"{button_key}_done"]:
        placeholder = st.empty()
        for remaining in range(duration_sec, 0, -1):
            mins, secs = divmod(remaining, 60)
            placeholder.markdown(f"**{message} — {mins:02d}:{secs:02d}**")
            time.sleep(1)
        placeholder.empty()
        st.session_state[f"{button_key}_done"] = True

    # Show button only after countdown done
    return st.button(button_label, key=button_key)

def countdown_with_form(message, duration_sec, form_key, input_key, submit_label="➤"):
    """
    Shows a countdown before revealing a form with text input + submit.
    Returns the user input if submitted, else None.
    """
    if f"{form_key}_done" not in st.session_state:
        st.session_state[f"{form_key}_done"] = False

    if not st.session_state[f"{form_key}_done"]:
        placeholder = st.empty()
        for remaining in range(duration_sec, 0, -1):
            mins, secs = divmod(remaining, 60)
            placeholder.markdown(f"**{message} — {mins:02d}:{secs:02d}**")
            time.sleep(1)
        placeholder.empty()
        st.session_state[f"{form_key}_done"] = True

    # Show form after countdown done
    if st.session_state[f"{form_key}_done"]:
        with st.form(form_key, clear_on_submit=True):
            cols = st.columns([4, 0.5])
            user_input = cols[0].text_input("", key=input_key, label_visibility="collapsed")
            send = cols[1].form_submit_button(submit_label)
            if send and user_input:
                return user_input
    return None
# VERSION 1
def stream_llm_api(history):
    """
    Streams assistant response from LLM API, chunk by chunk.
    Yields text in real time for display in st.chat_message container.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": history,
        "stream": True
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        with requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    try:
                        obj = json.loads(line)
                        chunk = obj.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
                    except Exception as parse_err:
                        print("⚠️ Chunk parsing error:", parse_err)
                        continue
    except requests.RequestException as e:
        st.markdown("LLM stream error: {e}")
        # raise RuntimeError(f"LLM stream failed: {e}")


def stream_to_llm_chat(history, container):
    """
    Wrapper for stream_llm_api that mimics the interface of stream_to_llm.
    Sends user+system history to LLM and streams response into the container.
    """
    assistant_plain = ""
    try:
        current = "<div class='assistant-msg'><b>🤖</b> "
        for chunk in stream_llm_api(history):
            assistant_plain += chunk
            current += chunk

            html_code = f"""<div id="chat" class='scrollbox'>{st.session_state.chat_html + current}</div>
                            <script>
                                var chat = document.getElementById("chat");
                                chat.scrollTop = chat.scrollHeight;
                            </script>"""
            container.markdown(html_code, unsafe_allow_html=True)

        current += "</div>"
        rawChat = current.replace("<div class='assistant-msg'><b>🤖</b> ", "").replace("</div>", "")
        st.session_state.chat_html += current
        st.session_state.chat_history.append({"role": "assistant", "content": rawChat})
        return assistant_plain

    except Exception as e:
        err = f"<div class='assistant-msg'><b>🤖</b> Error: {e}</div>"
        st.session_state.chat_html += err
        html_code = f"""<div id="chat" class='scrollbox'>{st.session_state.chat_html}</div>
                        <script>
                            var chat = document.getElementById("chat");
                            chat.scrollTop = chat.scrollHeight;
                        </script>"""
        container.markdown(html_code, unsafe_allow_html=True)
        return False


def render_v2_page():
    render_back_button("v2")
    st.title("Doctor Specialist Recommender")
    st.subheader("Version 2 - Pre-Quiz Explanation Flow")
    st.divider()

    scenario = render_scenario(0)
    selected_symptoms = render_symptom_selector()

    if st.button("Predict"):
        if handle_prediction(selected_symptoms):
            reset_lock_timer()
            reset_prequiz_states()

            st.session_state.v2_quiz_index = 0
            st.session_state.v2_selected_options = []
            st.session_state.v2_quiz_done = False
            st.session_state.v2_chat_history_per_q = {}
            st.session_state.v2_sent_system_prompt = {}
            st.session_state.v2_initial_radio_set = {}
            st.session_state.v2_input_used = {}
            st.rerun()

    if not st.session_state.prediction_ready:
        render_spacer()

    if st.session_state.prediction_ready:
        display_results(
            st.session_state.selected_symptoms_clean,
            st.session_state.top_classes,
            st.session_state.top_probs,
            st.session_state.specialists,
            st.session_state.specialists_pb,
            showStalemate=False
        )

        questions = load_prequiz_questions(scenario)
        idx = st.session_state.get("v2_quiz_index", 0)
        total = len(questions)
        if idx < total:
            for i in range(idx + 1):
                render_v2_quiz_flow(questions, i, scenario)

def make_quiz_system_prompt(question, options, correct_index, selected_symptoms, top_classes, top_probs, specialists, specialist_probs, scenario):
    prompt = f"""
        You are helping a user understand a medical diagnosis recommender system.
        Here is a patient scenario: "{scenario}"
        The user selected symptoms: {', '.join(selected_symptoms)}
        The model predicted diseases: {', '.join(top_classes)} with probabilities: {top_probs}
        The recommended specialists were: {', '.join(specialists)} with confidences: {specialist_probs}

        Now the user is answering this question:
        "{question}"
        Options:
        1. {options[0]}
        2. {options[1]}
        3. {options[2]}
        4. {options[3]}

        The correct answer is option {correct_index}, but do not reveal this unless asked.
        Your goal is to help the user understand how the system would reason about the question and its options.
        Keep your responses concise, clear, and focused on the question.
        Do not provide lengthy explanations.
        """
    return prompt.strip()

def render_v2_quiz_flow(questions, idx, scenario):
    question = questions[idx]
    qid = question['id']
    st.markdown(f"#### Q{idx+1}: {question['prompt']}")

    options = [question['opt1'], question['opt2'], question['opt3'], question['opt4']]

    selected_option_key = f"selected_option_q_{qid}"
    radio_key = f"option_radio_q_{qid}"

    if selected_option_key not in st.session_state:
        st.session_state[selected_option_key] = None

    prev_selected = st.session_state[selected_option_key]

    selected = st.radio(
        "Select your answer:",
        options,
        index=options.index(prev_selected) if prev_selected in options else None,
        key=radio_key
    )

    if selected != prev_selected:
        st.session_state[selected_option_key] = selected
        chosen_index = options.index(selected) + 1 if selected in options else None
        if chosen_index is not None:
            selected_text = options[chosen_index - 1]
            system_selected_msg = {
                "role": "system",
                "content": f"User's most recent or currently selected option {chosen_index}: \"{selected_text}\"."
            }
            # Make sure chat_history exists before appending
            if qid not in st.session_state.v2_chat_history_per_q:
                st.session_state.v2_chat_history_per_q[qid] = []
            st.session_state.v2_chat_history_per_q[qid].append(system_selected_msg)

    correct = question['correct_index']
    chosen = options.index(selected) + 1 if selected in options else None

    if chosen:
        if chosen == correct:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Incorrect.")

    # Initialize quiz session state dicts if needed
    if "v2_chat_history_per_q" not in st.session_state:
        st.session_state.v2_chat_history_per_q = {}
    if "v2_sent_system_prompt" not in st.session_state:
        st.session_state.v2_sent_system_prompt = {}
    if "v2_input_used" not in st.session_state:
        st.session_state.v2_input_used = {}

    if qid not in st.session_state.v2_chat_history_per_q:
        st.session_state.v2_chat_history_per_q[qid] = []

    chat_history = st.session_state.v2_chat_history_per_q[qid]

    if qid not in st.session_state.v2_sent_system_prompt:
        system_prompt = make_quiz_system_prompt(
            question['prompt'], options, correct,
            st.session_state.selected_symptoms_clean,
            st.session_state.top_classes,
            st.session_state.top_probs,
            st.session_state.specialists,
            st.session_state.specialists_pb,
            scenario
        )
        chat_history.append({"role": "system", "content": system_prompt})
        st.session_state.v2_sent_system_prompt[qid] = True

    # Render all messages excluding system
    for msg in chat_history:
        if msg["role"] != "system":
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

    # --- Add streaming flag init per question ---
    streaming_flag_key = f"v2_is_streaming_{qid}"
    if streaming_flag_key not in st.session_state:
        st.session_state[streaming_flag_key] = False

    # Only render form and collect input if this is the active quiz question AND not currently streaming
    if idx == st.session_state.v2_quiz_index:
        if st.session_state[streaming_flag_key]:
            # Stream assistant response (no form)
            try:
                with st.chat_message("assistant"):
                    response_container = st.empty()
                    assistant_text = ""
                    for chunk in stream_llm_api(chat_history):
                        assistant_text += chunk
                        response_container.markdown(assistant_text + "▌")
                    response_container.markdown(assistant_text)
                    chat_history.append({"role": "assistant", "content": assistant_text})
                    log_message("assistant", assistant_text)
                st.session_state[streaming_flag_key] = False  # done streaming
                st.rerun()  # rerun so form can show next run
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"LLM error: {e}")
                st.session_state[streaming_flag_key] = False
        else:
            if st.session_state.get("v2_show_final_chat", False):
                # If final chat is showing, skip rendering per-question input form
                pass
            else:
                # Not streaming → show form to collect user input
                user_input = countdown_with_form(
                    message="Please read the text question carefully before answering.",
                    duration_sec=st.session_state.get("COOLDOWN_TIME_SHORT", COOLDOWN_TIME_SHORT),
                    form_key=f"form_q_{qid}",
                    input_key=f"input_q_{qid}"
                )

                if user_input:
                    chat_history.append({"role": "user", "content": user_input})
                    with st.chat_message("user"):
                        st.markdown(user_input)
                    # Set streaming flag to true to trigger streaming on next rerun
                    st.session_state[streaming_flag_key] = True
                    st.rerun()

        # Show Next / Finish button logic
        if st.session_state.v2_quiz_index < len(questions) - 1:
            # For all but last question, show normal Next button
            if st.button("Next", key=f"next_btn_{idx}"):
                st.session_state.v2_selected_options.append({
                    "question_id": question['id'],
                    "selected": chosen,
                    "correct": correct
                })
                st.session_state.v2_quiz_index += 1
                st.rerun()

        else:
            # Last question: control when to show final chatbot form with a flag
            if not st.session_state.get("v2_show_final_chat", False):
                # Show a "Finish Quiz" button first
                if st.button("Finish Quiz", key=f"finish_btn_{idx}"):
                    # Save last question answer before finishing
                    st.session_state.v2_selected_options.append({
                        "question_id": question['id'],
                        "selected": chosen,
                        "correct": correct
                    })
                    st.session_state.v2_show_final_chat = True
                    st.rerun()

            else:
                # Show final chatbot input after "Finish Quiz" pressed
                st.markdown("---")
                st.markdown("#### 📝 Do you have any other questions?")
                system_prompt = make_system_prompt(
                    st.session_state.selected_symptoms_clean,
                    st.session_state.top_classes,
                    st.session_state.top_probs,
                    st.session_state.specialists,
                    [round(x * 100, 2) for x in st.session_state.specialists_pb],
                    scenario
                )
                if "final_chat_history" not in st.session_state:
                    st.session_state.final_chat_history = [
                        {"role": "system", "content": system_prompt}
                    ]
                if "final_streaming" not in st.session_state:
                    st.session_state.final_streaming = False

                # Render final chat transcript so far
                for msg in st.session_state.final_chat_history:
                    if msg["role"] == "system":
                        continue
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                # Stream assistant if streaming flag set
                if st.session_state.final_streaming:
                    with st.chat_message("assistant"):
                        response_container = st.empty()
                        assistant_text = ""
                        for chunk in stream_llm_api(st.session_state.final_chat_history):
                            assistant_text += chunk
                            response_container.markdown(assistant_text + "▌")
                        response_container.markdown(assistant_text)

                    st.session_state.final_streaming = False
                    st.session_state.final_chat_history.append({"role": "assistant", "content": assistant_text})
                    st.rerun()

                else:
                    # Show freeform input form
                    user_input = countdown_with_form(
                        message="Please wait",
                        duration_sec=st.session_state.get("COOLDOWN_TIME_LONG", COOLDOWN_TIME_LONG),
                        form_key="final_freeform_form",
                        input_key="final_freeform_input"
                    )

                    if user_input:
                        st.session_state.final_chat_history.append({"role": "user", "content": user_input})
                        with st.chat_message("user"):
                            st.markdown(user_input)
                        st.session_state.final_streaming = True
                        st.rerun()

    st.session_state.v2_chat_history_per_q[qid] = chat_history

def load_prequiz_questions(scenario):
    if "v2_quiz_questions" not in st.session_state:
        patient_name = ("").join(scenario.split(" ")[:2])
        resp = supabase.table("prequiz_questions") \
            .select("id, prompt, opt1, opt2, opt3, opt4, correct_index") \
            .eq("patient_name", patient_name) \
            .order("id", desc=False) \
            .limit(3) \
            .execute()
        st.session_state.v2_quiz_questions = resp.data

    return st.session_state.v2_quiz_questions

# VERSION 1
def render_v1_page():
    render_back_button("v1")
    st.title("Doctor Specialist Recommender")
    st.subheader("Version 1 - Prediction Model with AI Chatbot")
    st.divider()

    scenario = render_scenario(2)
    selected_symptoms = render_symptom_selector()
    

    if st.button("Predict"):
        if handle_prediction(selected_symptoms):
            st.session_state.followup_idx = 1
            reset_lock_timer()
            st.rerun()
    if not st.session_state.prediction_ready:
        render_spacer()

    if st.session_state.prediction_ready:
        render_v1_prediction_results()
        render_v1_explanation_flow(scenario)

def render_v1_prediction_results():
    selected_symptoms_clean = st.session_state.selected_symptoms_clean
    top_classes = st.session_state.top_classes
    top_probs = st.session_state.top_probs
    specialist = st.session_state.specialists
    specialist_prob = [round(x * 100, 2) for x in st.session_state.specialists_pb]

    sym_str = ", ".join(selected_symptoms_clean[:-1]) + f", and {selected_symptoms_clean[-1]}" if len(selected_symptoms_clean) > 1 else selected_symptoms_clean[0]

    st.success(f"Your symptoms {sym_str} suggest that you might have the following diseases with respective confidence as shown:")

    st.markdown("Top likely diseases are:")
    for i in range(3):
        st.markdown(f"**{i+1}) {top_classes[i]}** ({round(top_probs[i]*100, 2)} % confidence)")

    st.info(f"For these diseases, our **Specialist Recommendation** model suggests: \n\n- **{specialist[0]}** ({specialist_prob[0]} % confidence)\n- **{specialist[1]}** ({specialist_prob[1]} % confidence)")
    st.markdown("---")


def render_v1_explanation_flow(scenario):
    explain_container = st.empty()
    patient_name = scenario.split(" ")[0] if scenario.split(" ") else "the patient"

    questions = [
        "Can you explain how the system takes symptoms and produces the results?",
        f"How did {patient_name} get these specific recommendations?",
        f"What if {patient_name} had different symptoms, how would that change the results?",
    ]

    if DEBUG:
        questions = ['Hi', 'Thank you','ok']
    if st.session_state.show_explain_option:
        # explain_container.markdown("**Do you want a more detailed explanation?**")
        if countdown_with_button("Please read the results carefully", st.session_state.get("COOLDOWN_TIME_SHORT", COOLDOWN_TIME_SHORT), questions[0], "explain_btn"):
            st.session_state.show_explain_option = False
            st.session_state.explain_clicked = True
            explain_container.empty()
            st.rerun()

    if st.session_state.explain_clicked:
        st.markdown("### 💬 Explanation and Follow-ups")

    chat_box = st.empty()
    # if st.session_state.initial_prompt_sent or st.session_state.explain_clicked:
    #     render_chat_styles(chat_box)

    if st.session_state.explain_clicked:
        chat_box = start_llm_chat(scenario, questions)

    if st.session_state.initial_prompt_sent:
        continue_llm_chat(questions)

# def render_chat_transcript():
#     """Render full transcript from session_state.chat_history (exclude system)."""
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
#     for msg in st.session_state.chat_history:
#         if msg.get("role") == "system":
#             continue
#         with st.chat_message(msg.get("role")):
#             st.markdown(msg.get("content"))
    
    
# def render_chat_styles(chat_box):
#     st.markdown("""
#     <style>
#         .scrollbox {
#             border: 1px solid #ddd;
#             padding: 10px;
#             border-radius: 10px;
#             background-color: #f9f9f9;
#             font-size: 0.95rem;
#         }
#         .user-msg { color: #0b0c0c; margin-bottom: 0.5em; background-color: gainsboro; padding-top: 10px; padding-bottom: 10px;}
#         .assistant-msg { color: #009966; margin-bottom: 1em; }
#     </style>
#     """, unsafe_allow_html=True)
#     chat_box.markdown("<div id='chat' class='scrollbox'></div>", unsafe_allow_html=True)


# def start_llm_chat(scenario, questions, chat_box):
#     selected_symptoms_clean = st.session_state.selected_symptoms_clean
#     top_classes = st.session_state.top_classes
#     top_probs = st.session_state.top_probs
#     specialist = st.session_state.specialists
#     specialist_prob = [round(x * 100, 2) for x in st.session_state.specialists_pb]

#     system_prompt = make_system_prompt(selected_symptoms_clean, top_classes, top_probs, specialist, specialist_prob, scenario)
#     st.session_state.chat_history = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user",   "content": questions[0]}
#     ]    

#     st.session_state.initial_prompt_sent = True
#     st.session_state.explain_clicked = False

    
#     with st.chat_message("user"):
#         st.markdown(questions[0])
#     try:
#         with st.chat_message("assistant"):
#             response_container = st.empty()
#             assistant_text = ""
#             for chunk in stream_llm_api(st.session_state.chat_history):
#                 assistant_text += chunk
#                 response_container.markdown(assistant_text + "▌")
#             response_container.markdown(assistant_text)
#             st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
#             log_message("user", questions[0])
#             log_message("assistant", assistant_text)
#     except Exception as e:
#         with st.chat_message("assistant"):
#             st.error(f"LLM error: {e}")
#     return chat_box

# def continue_llm_chat(questions, chat_box):
#     chat_box.empty()

#     idx = st.session_state.followup_idx

#     # Only replay transcript after Q1 → Q2 transition
#     if idx > 1:
#         render_chat_transcript()
        
#     def ask_and_advance(i):
#         q = questions[i]
#         st.session_state.chat_history.append({"role": "user", "content": q})

#         with st.chat_message("user"):
#             st.markdown(q)
#         try:
#             with st.chat_message("assistant"):
#                 chat_box = st.empty()
#                 assistant_text = ""
#                 for chunk in stream_llm_api(st.session_state.chat_history):
#                     assistant_text += chunk
#                     chat_box.markdown(assistant_text + "▌")
#                 chat_box.markdown(assistant_text)
#                 st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
#                 log_message("user", q)
#                 log_message("assistant", assistant_text)

#         except Exception as e:
#             with st.chat_message("assistant"):
#                 st.error(f"LLM error: {e}")


#         st.session_state.followup_idx += 1

#     if idx < len(questions):
#         if countdown_with_button(
#             message="Please read the generated text carefully",
#             duration_sec=COOLDOWN_TIME_LONG,
#             button_label=questions[idx],
#             button_key=f"followup_btn_{idx}"
#         ):
#             ask_and_advance(idx)
#             st.rerun()

#     if idx >= len(questions):
#         user_input = countdown_with_form(
#             message="Please read carefully before interacting with the chatbot",
#             duration_sec=COOLDOWN_TIME_LONG,
#             form_key="freeform_followup",
#             input_key="freeform_input"
#         )
#         if user_input:
#             st.session_state.chat_history.append({"role": "user", "content": user_input})
#             with st.chat_message("user"):
#                 st.markdown(user_input)
#             try:
#                 with st.chat_message("assistant"):
#                     response_container = st.empty()
#                     assistant_text = ""
#                     for chunk in stream_llm_api(st.session_state.chat_history):
#                         assistant_text += chunk
#                         response_container.markdown(assistant_text + "▌")
#                     response_container.markdown(assistant_text)
#                     st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
#                     log_message("user", user_input)
#                     log_message("assistant", assistant_text)

#             except Exception as e:
#                 with st.chat_message("assistant"):
#                     st.error(f"LLM error: {e}")

def render_chat_transcript():
    """Render full transcript from session_state.chat_history (exclude system)."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def start_llm_chat(scenario, questions):
    # Build system prompt & initialize history
    system_prompt = make_system_prompt(
        st.session_state.selected_symptoms_clean,
        st.session_state.top_classes,
        st.session_state.top_probs,
        st.session_state.specialists,
        [round(x * 100, 2) for x in st.session_state.specialists_pb],
        scenario
    )

    st.session_state.chat_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": questions[0]}
    ]
    st.session_state.initial_prompt_sent = True
    st.session_state.explain_clicked = False
    st.session_state.followup_idx = 1  # reset index

    # Always render what’s in history first
    render_chat_transcript()

    # Stream only for the active question (first one)
    with st.chat_message("assistant"):
        response_container = st.empty()
        assistant_text = ""
        for chunk in stream_llm_api(st.session_state.chat_history):
            assistant_text += chunk
            response_container.markdown(assistant_text + "▌")
        response_container.markdown(assistant_text)

    # Append the streamed message to history (so it shows next rerun)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
    log_message("user", questions[0])
    log_message("assistant", assistant_text)
    st.rerun()  # force rerun so transcript now includes it

def continue_llm_chat(questions):
    idx = st.session_state.followup_idx

    # Initialize streaming flag if not set
    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False

    # Render the chat transcript so far
    render_chat_transcript()

    # If streaming is active, stream the assistant's reply now
    if st.session_state.is_streaming:
        with st.chat_message("assistant"):
            response_container = st.empty()
            assistant_text = ""
            for chunk in stream_llm_api(st.session_state.chat_history):
                assistant_text += chunk
                response_container.markdown(assistant_text + "▌")
            response_container.markdown(assistant_text)
        
        # Update streaming flag and rerun after done
        st.session_state.is_streaming = False
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
        st.rerun()

    # Only if not streaming, render buttons/forms for next user input
    else:
        def ask_and_advance(q):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.session_state.is_streaming = True  # Set streaming flag to True for next rerun
            st.session_state.followup_idx += 1
            st.rerun()

        # Scripted questions buttons
        if idx < len(questions):
            if countdown_with_button(
                message="Please read the generated text carefully",
                duration_sec=st.session_state.get("COOLDOWN_TIME_LONG", COOLDOWN_TIME_LONG),
                button_label=questions[idx],
                button_key=f"followup_btn_{idx}"
            ):
                ask_and_advance(questions[idx])
        # Free-form input
        else:
            user_input = countdown_with_form(
                message="Please read carefully before interacting with the chatbot",
                duration_sec=st.session_state.get("COOLDOWN_TIME_LONG", COOLDOWN_TIME_LONG),
                form_key="freeform_followup",
                input_key="freeform_input"
            )
            if user_input:
                ask_and_advance(user_input)

if st.session_state.page == "home":
    render_home_page()

elif st.session_state.page == "v2":
    render_v2_page()

elif st.session_state.page == "v1":
    render_v1_page()
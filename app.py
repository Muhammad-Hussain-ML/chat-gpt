import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# --- Configure Google Generative AI ---
if not api_key:
    st.error("GEMINI_API_KEY not found in environment. Please set it in your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# --- Streamlit UI setup ---
st.set_page_config(page_title="Gemini Chat", page_icon="💬", layout="wide")
st.title("💬 Gemini Chat Interface")

# --- Model and Temperature Selection ---
# Corrected and updated model names for Gemini 1.5 and older stable models
model_options = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
selected_model = st.sidebar.selectbox("🧠 Choose a Gemini model:", model_options)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# --- Initialize the GenerativeModel object once per session ---
# This ensures the model is re-initialized only if the selected model or temperature changes
# or if it's the first run.
if "llm_model" not in st.session_state or \
   st.session_state.get("last_selected_model") != selected_model or \
   st.session_state.get("last_temperature") != temperature:

    st.session_state.llm_model = genai.GenerativeModel(
        model_name=selected_model, 
        generation_config={"temperature": temperature}
    )
    st.session_state.last_selected_model = selected_model
    st.session_state.last_temperature = temperature

# Assign the model from session state for use in the script
llm_model = st.session_state.llm_model

# --- Chat Memory and Session Management ---
MAX_HISTORY_LENGTH = 30  # Adjust as needed for context window management

# Initialize chat_history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Display existing chat history ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        # Ensure 'parts' is iterable and joinable, even if it's just a string
        content = msg["parts"] if isinstance(msg["parts"], list) else [msg["parts"]]
        st.markdown("".join(content))

# --- Handle user input ---
user_input = st.chat_input("Ask something...")
if user_input:
    # Add user input to history and display it immediately
    st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build the prompt including chat history (with truncation)
    # This prompt is sent to the model for content generation
    prompt_parts = []
    for turn in st.session_state.chat_history[-MAX_HISTORY_LENGTH:]: 
        role = "user" if turn["role"] == "user" else "model" # Gemini expects 'user' or 'model' roles
        content = "\n".join(turn["parts"]).strip()
        prompt_parts.append({"role": role, "parts": [content]})

    # The last turn is always the user's current input, so we don't need to add "Assistant:"
    # The `generate_content` method with a list of dicts handles the turn-taking implicitly.

    with st.chat_message("model"):
        try:
            full_response = ""
            # Pass the list of prompt parts directly to generate_content
            # The temperature is already set on llm_model, no need to pass generation_config again.
            response = llm_model.generate_content(prompt_parts, stream=True) 

            placeholder = st.empty()
            for chunk in response:
                # Handle cases where chunk.text might be None or empty
                if chunk.text:
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌") # Typing effect
            placeholder.markdown(full_response) # Final response without cursor

            # Save model's response to chat history
            st.session_state.chat_history.append({"role": "model", "parts": [full_response]})

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            # Optionally, remove the last user message from history if generation failed
            # st.session_state.chat_history.pop() 





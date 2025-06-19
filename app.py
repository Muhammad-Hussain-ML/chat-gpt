import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
st.write(api_key)
# --- Configure Google Generative AI ---
if not api_key:
    st.error("GOOGLE_API_KEY not found in environment. Please set it in your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# --- Streamlit UI setup ---
st.set_page_config(page_title="Gemini Chat", page_icon="💬", layout="wide")
st.title("💬 Gemini Chat Interface (ChatGPT-style)")

# --- Model Selection ---
model_options = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash"
]
selected_model = st.sidebar.selectbox("🧠 Choose a Gemini model:", model_options)

# --- Chat Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- File Upload ---
uploaded_file = st.file_uploader("📎 Upload image or document (optional)", type=["jpg", "jpeg", "png", "pdf", "txt"])
file_part = None
if uploaded_file:
    file_part = genai.types.Part.from_data(
        data=uploaded_file.read(),
        mime_type=uploaded_file.type
    )
    st.success(f"Uploaded: {uploaded_file.name}")

# --- Show chat history ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown("".join(msg["parts"]))

# --- Handle user input ---
user_input = st.chat_input("Ask something...")
if user_input:
    # Add user message to history
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "parts": [user_input]})

    # Create model and chat session
    model = genai.GenerativeModel(model_name=selected_model)
    chat = model.start_chat(history=st.session_state.chat_history)

    # Send message
    with st.chat_message("model"):
        try:
            response = chat.send_message(user_input, files=[file_part] if file_part else None, stream=True)
            output = ""
            for chunk in response:
                output += chunk.text
                st.markdown(output + "▌")  # typing effect
            st.markdown(output)
            st.session_state.chat_history.append({"role": "model", "parts": [output]})
        except Exception as e:
            st.error(f"Error: {e}")

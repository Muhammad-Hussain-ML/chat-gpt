import streamlit as st
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- 1. CONFIGURATION & SESSION STATE ---
load_dotenv()
st.set_page_config(
    page_title="Gemini Next-Gen AI", 
    page_icon="🤖", 
    layout="centered"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .source-box { 
        background-color: #f0f2f6; 
        padding: 10px; 
        border-radius: 10px; 
        font-size: 0.85rem; 
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. SIDEBAR SETTINGS ---
with st.sidebar:
    st.title("🚀 Gemini Settings")
    
    # API Key Handling (Prioritize Sidebar over .env)
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key", value=env_key, type="password")
    
    st.divider()
    
    # Model Selection (Current 2025 Stable/Preview Models)
    model_map = {
        "Gemini 2.0 Flash (Fastest)": "gemini-2.0-flash",
        "Gemini 2.0 Pro (Most Powerful)": "gemini-2.0-pro-exp-02-05",
        "Gemini 1.5 Pro (Stable)": "gemini-1.5-pro",
    }
    selected_label = st.selectbox("Brain Model", options=list(model_map.keys()))
    selected_model = model_map[selected_label]
    
    st.divider()
    
    # Tool Toggles
    st.subheader("Capabilities")
    enable_search = st.checkbox("Google Search (Grounding)", value=True, help="AI will search the web for real-time info")
    enable_code = st.checkbox("Code Execution", value=True, help="AI can write and run Python to solve problems")
    
    st.divider()
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- 3. CLIENT INITIALIZATION ---
if not api_key:
    st.info("Please enter your Gemini API Key in the sidebar to start.", icon="🔑")
    st.stop()

# New SDK Client
client = genai.Client(api_key=api_key)

# --- 4. DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. CHAT LOGIC ---
if prompt := st.chat_input("Ask me anything..."):
    # Add User Message to UI and State
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare Gemini Tools
    tools = []
    if enable_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    if enable_code:
        tools.append(types.Tool(code_execution=types.CodeExecution()))

    # Build History for the SDK
    # Gemini 2.0 expects 'user' and 'model' roles
    history = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

    # Generate Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        grounding_metadata = None
        
        try:
            # Using the new streaming method
            stream = client.models.generate_content_stream(
                model=selected_model,
                contents=history,
                config=types.GenerateContentConfig(
                    tools=tools,
                    system_instruction="You are a professional AI. Use tools when needed. Format code with backticks.",
                    temperature=0.7
                )
            )

            # Process Stream Chunks
            for chunk in stream:
                # 1. Handle Text Content
                if chunk.text:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
                
                # 2. Capture Grounding/Search Metadata (if any)
                if chunk.grounding_metadata:
                    grounding_metadata = chunk.grounding_metadata

            # Final Text Update
            response_placeholder.markdown(full_response)

            # 3. Handle Grounding Sources UI
            if grounding_metadata and grounding_metadata.search_entry_point:
                with st.expander("🌐 Verified Sources (Google Search)"):
                    # Render the Google Search UI snippet
                    st.components.v1.html(
                        grounding_metadata.search_entry_point.html_content, 
                        height=150, 
                        scrolling=True
                    )

            # 4. Handle Code Execution Parts (Advanced UI)
            # The SDK allows checking if the model actually ran code
            for part in chunk.candidates[0].content.parts:
                if part.executable_code:
                    with st.expander("🛠️ View Generated Code"):
                        st.code(part.executable_code.code, language="python")
                if part.code_execution_result:
                    with st.expander("📤 Execution Output"):
                        st.info(part.code_execution_result.output)

            # Save Assistant Message
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")




# import streamlit as st
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv

# # --- Load environment variables ---
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")

# # --- Configure Google Generative AI ---
# if not api_key:
#     st.error("GEMINI_API_KEY not found in environment. Please set it in your .env file.")
#     st.stop()

# genai.configure(api_key=api_key)

# # --- Streamlit UI setup ---
# st.set_page_config(page_title="Gemini Chat", page_icon="💬", layout="wide")
# st.title("💬 Gemini Chat Interface")

# # --- Model and Temperature Selection ---
# # Corrected and updated model names for Gemini 1.5 and older stable models
# model_options = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
# selected_model = st.sidebar.selectbox("🧠 Choose a Gemini model:", model_options)
# temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# # --- Initialize the GenerativeModel object once per session ---
# # This ensures the model is re-initialized only if the selected model or temperature changes
# # or if it's the first run.
# if "llm_model" not in st.session_state or \
#    st.session_state.get("last_selected_model") != selected_model or \
#    st.session_state.get("last_temperature") != temperature:

#     st.session_state.llm_model = genai.GenerativeModel(
#         model_name=selected_model, 
#         generation_config={"temperature": temperature}
#     )
#     st.session_state.last_selected_model = selected_model
#     st.session_state.last_temperature = temperature

# # Assign the model from session state for use in the script
# llm_model = st.session_state.llm_model

# # --- Chat Memory and Session Management ---
# MAX_HISTORY_LENGTH = 30  # Adjust as needed for context window management

# # Initialize chat_history if it doesn't exist
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # --- Display existing chat history ---
# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         # Ensure 'parts' is iterable and joinable, even if it's just a string
#         content = msg["parts"] if isinstance(msg["parts"], list) else [msg["parts"]]
#         st.markdown("".join(content))

# # --- Handle user input ---
# user_input = st.chat_input("Ask something...")
# if user_input:
#     # Add user input to history and display it immediately
#     st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     # Build the prompt including chat history (with truncation)
#     # This prompt is sent to the model for content generation
#     prompt_parts = []
#     for turn in st.session_state.chat_history[-MAX_HISTORY_LENGTH:]: 
#         role = "user" if turn["role"] == "user" else "model" # Gemini expects 'user' or 'model' roles
#         content = "\n".join(turn["parts"]).strip()
#         prompt_parts.append({"role": role, "parts": [content]})

#     # The last turn is always the user's current input, so we don't need to add "Assistant:"
#     # The `generate_content` method with a list of dicts handles the turn-taking implicitly.

#     with st.chat_message("model"):
#         try:
#             full_response = ""
#             # Pass the list of prompt parts directly to generate_content
#             # The temperature is already set on llm_model, no need to pass generation_config again.
#             response = llm_model.generate_content(prompt_parts, stream=True) 

#             placeholder = st.empty()
#             for chunk in response:
#                 # Handle cases where chunk.text might be None or empty
#                 if chunk.text:
#                     full_response += chunk.text
#                     placeholder.markdown(full_response + "▌") # Typing effect
#             placeholder.markdown(full_response) # Final response without cursor

#             # Save model's response to chat history
#             st.session_state.chat_history.append({"role": "model", "parts": [full_response]})

#         except Exception as e:
#             st.error(f"An unexpected error occurred: {e}")
#             # Optionally, remove the last user message from history if generation failed
#             # st.session_state.chat_history.pop() 





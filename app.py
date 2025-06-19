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
model_options = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-flash"]
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


# --- Clear Chat History button ---
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun() # Rerun to clear the displayed messages immediately








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
# model_options = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
# selected_model = st.sidebar.selectbox("🧠 Choose a Gemini model:", model_options)
# temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# # --- Chat Memory and Session Management ---
# MAX_HISTORY_LENGTH = 30  # Adjust as needed
# if "chat" not in st.session_state or st.session_state.get("last_model") != selected_model:
#     st.session_state.chat = genai.GenerativeModel(model_name=selected_model, generation_config={"temperature": temperature}).start_chat()
#     st.session_state.last_model = selected_model

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []


# # --- Show chat history ---
# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.markdown("".join(msg["parts"]))


# # --- Handle user input ---
# user_input = st.chat_input("Ask something...")
# if user_input:
#     st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     chat = st.session_state.chat

#     # Build the prompt including chat history (with truncation)
#     prompt = ""
#     for turn in st.session_state.chat_history[-MAX_HISTORY_LENGTH:]: 
#         role = "User" if turn["role"] == "user" else "Assistant"
#         content = "\n".join(turn["parts"]).strip()
#         prompt += f"{role}:\n{content}\n\n"
#     prompt += "Assistant:\n"

#     with st.chat_message("model"):
#         try:
#             st.write(f"prompt : {prompt}")
#             response = chat.send_message(content=prompt, stream=True)
#             full_response = ""
#             placeholder = st.empty()
#             for chunk in response:
#                 full_response += chunk.text
#                 placeholder.markdown(full_response + "▌")
#             placeholder.markdown(full_response)
#             st.session_state.chat_history.append({"role": "model", "parts": [full_response]})

#         except Exception as e:
#             st.error(f"An unexpected error occurred: {e}")


# # --- Clear Chat History button ---
# if st.button("Clear Chat History"):
#     st.session_state.chat_history = []
#     st.session_state.chat = genai.GenerativeModel(model_name=selected_model).start_chat()





# import streamlit as st
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv
# import PyPDF2
# from PIL import Image
# import pytesseract  # For OCR (Image to Text)


# # --- Load environment variables ---
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")  # Make sure your API key is set

# # --- Configure Google Generative AI ---
# if not api_key:
#     st.error("GEMINI_API_KEY not found in environment. Please set it in your .env file.")
#     st.stop()

# genai.configure(api_key=api_key)

# # --- Streamlit UI setup ---
# st.set_page_config(page_title="Gemini Chat", page_icon="💬", layout="wide")
# st.title("💬 Gemini Chat Interface")

# # --- Model Selection ---
# model_options = [
#     "gemini-2.5-pro",
#     "gemini-2.5-flash",
#     "gemini-1.5-pro",
#     "gemini-1.5-flash"
# ]
# selected_model = st.sidebar.selectbox("🧠 Choose a Gemini model:", model_options)

# # --- Chat Memory and Session Management ---
# if "chat" not in st.session_state or st.session_state.get("last_model") != selected_model:
#     st.session_state.chat = genai.GenerativeModel(model_name=selected_model).start_chat()
#     st.session_state.last_model = selected_model 

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []


# # --- File Upload ---
# uploaded_file = st.file_uploader("📎 Upload file (optional)", type=["jpg", "jpeg", "png", "pdf", "txt"])
# file_content = None  # Store the processed file content

# if uploaded_file:
#     file_type = uploaded_file.type
#     if file_type == "application/pdf":
#         pdf_reader = PyPDF2.PdfReader(uploaded_file)
#         file_content = ""
#         for page in pdf_reader.pages:
#             file_content += page.extract_text()
#     elif file_type.startswith("image/"):  # Handle images (OCR)
#         try:
#             image = Image.open(uploaded_file)
#             file_content = pytesseract.image_to_string(image)
#         except Exception as e: # Handle potential OCR errors
#             st.error(f"OCR Error: {e}")
#             file_content = None # So it doesn't try to send bad data to Gemini
#     elif file_type == "text/plain":
#         file_content = uploaded_file.read().decode("utf-8")  # For text files
#     else:
#         st.warning("Unsupported file type. Please upload a PDF, image, or text file.")

#     if file_content:
#         st.success(f"Uploaded: {uploaded_file.name}")

# # --- Show chat history ---
# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.markdown("".join(msg["parts"]))

# # --- Handle user input ---
# user_input = st.chat_input("Ask something...")
# if user_input:
#     st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     chat = st.session_state.chat

#     contents = [user_input]
#     if file_content:
#         contents.append(file_content) 

#     with st.chat_message("model"):
#         try:
#             response = chat.send_message(content=contents, stream=True)
#             full_response = ""
#             placeholder = st.empty()
#             for chunk in response:
#                 full_response += chunk.text
#                 placeholder.markdown(full_response + "▌")
#             placeholder.markdown(full_response)
#             st.session_state.chat_history.append({"role": "model", "parts": [full_response]})
#         except genai.errors.APIError as e:
#             st.error(f"API Error: {e}")
#         except Exception as e:
#             st.error(f"An unexpected error occurred: {e}")



# # --- Clear File and Chat History buttons ---
# col1, col2 = st.columns(2)
# with col1:
#     if st.button("Clear Uploaded File"):
#         uploaded_file = None
#         file_content = None
# with col2:
#     if st.button("Clear Chat History"):
#         st.session_state.chat_history = []
#         st.session_state.chat = genai.GenerativeModel(model_name=selected_model).start_chat()














# import streamlit as st
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv

# # --- Load environment variables ---
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
# st.write(api_key)  # Optional: remove in production

# # --- Configure Google Generative AI ---
# if not api_key:
#     st.error("GEMINI_API_KEY not found in environment. Please set it in your .env file.")
#     st.stop()

# genai.configure(api_key=api_key)

# # --- Streamlit UI setup ---
# st.set_page_config(page_title="Gemini Chat", page_icon="💬", layout="wide")
# st.title("💬 Gemini Chat Interface (ChatGPT-style)")

# # --- Model Selection ---
# model_options = [
#     "gemini-2.5-pro",
#     "gemini-2.5-flash",
#     "gemini-1.5-pro",
#     "gemini-1.5-flash"
# ]
# selected_model = st.sidebar.selectbox("🧠 Choose a Gemini model:", model_options)

# # --- Chat Memory ---
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # --- File Upload ---
# uploaded_file = st.file_uploader("📎 Upload image or document (optional)", type=["jpg", "jpeg", "png", "pdf", "txt"])
# file_part = None
# if uploaded_file:
#     file_part = genai.types.Part.from_data(
#         data=uploaded_file.read(),
#         mime_type=uploaded_file.type
#     )
#     st.success(f"Uploaded: {uploaded_file.name}")

# # --- Show chat history ---
# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.markdown("".join(msg["parts"]))

# st.write(st.session_state.chat_history)
# # --- Handle user input ---
# user_input = st.chat_input("Ask something...")
# if user_input:
#     # Add user message to history
#     st.chat_message("user").markdown(user_input)
#     st.session_state.chat_history.append({"role": "user", "parts": [user_input]})

#     # Create model and chat session
#     model = genai.GenerativeModel(model_name=selected_model)
#     chat = model.start_chat(history=st.session_state.chat_history)

#     # Prepare contents with optional file
#     contents = [user_input]
#     if file_part:
#         contents.append(file_part)

#     # Send message and display streaming response
#     with st.chat_message("model"):
#         try:
#             response = chat.send_message(content=contents, stream=True)
#             output = ""
#             placeholder = st.empty()
#             for chunk in response:
#                 output += chunk.text
#                 placeholder.markdown(output + "▌")  # typing effect
#             placeholder.markdown(output)
#             st.session_state.chat_history.append({"role": "model", "parts": [output]})
#         except Exception as e:
#             st.error(f"Error: {e}")


import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
import os
import base64
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# ENV & CLIENT
# ─────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found. Please set it in your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Gemini Chat",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Gemini Chat Interface")

# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
STABLE_MODELS = {
    "⚡ Gemini 2.5 Flash  —  Best all-rounder (stable)":      "gemini-2.5-flash",
    "🧠 Gemini 2.5 Pro  —  Most capable (stable)":            "gemini-2.5-pro",
    "🪶 Gemini 2.5 Flash-Lite  —  Fastest & cheapest (stable)": "gemini-2.5-flash-lite",
}

PREVIEW_MODELS = {
    "🔬 Gemini 3.1 Pro  —  Most powerful overall (preview)":      "gemini-3.1-pro-preview",
    "🚀 Gemini 3 Flash  —  New-gen performance (preview)":        "gemini-3-flash-preview",
    "💨 Gemini 3.1 Flash-Lite  —  New-gen cheapest (preview)":   "gemini-3.1-flash-lite-preview",
}

ALL_MODELS       = {**STABLE_MODELS, **PREVIEW_MODELS}
MODEL_DISPLAY    = list(ALL_MODELS.keys())

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # ── Model picker ──────────────────────────
    st.markdown("**🧠 Model**")
    st.caption("Stable = production-ready · Preview = cutting-edge, may have rate limits")

    selected_display = st.selectbox(
        "Model",
        options=MODEL_DISPLAY,
        index=0,
        label_visibility="collapsed",
    )
    selected_model = ALL_MODELS[selected_display]

    if selected_display in STABLE_MODELS:
        st.success(f"✅ Stable  |  `{selected_model}`")
    else:
        st.warning(f"⚠️ Preview  |  `{selected_model}`")

    st.divider()

    # ── Temperature ───────────────────────────
    st.markdown("**🌡️ Temperature**")
    st.caption("0 = deterministic · 1 = balanced (default) · 2 = very creative")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        label_visibility="collapsed",
    )

    # Live hint based on value
    if temperature == 0.0:
        st.info("🎯 Deterministic — always picks the most likely word. Best for math, code, facts.")
    elif temperature < 0.5:
        st.info("📌 Very focused — minimal variation. Good for summarisation and precise Q&A.")
    elif temperature <= 1.0:
        st.info("⚖️ Balanced — works well for most tasks.")
    elif temperature <= 1.5:
        st.info("🎨 Creative — more variation. Good for writing and brainstorming.")
    else:
        st.info("🌀 Very random — experimental and unpredictable output.")

    if selected_display in PREVIEW_MODELS and temperature != 1.0:
        st.warning("⚠️ Google recommends keeping temperature at 1.0 for Gemini 3 models.")

    st.divider()

    # ── Tools ─────────────────────────────────
    st.markdown("**🛠️ Tools**")

    use_search = st.toggle(
        "🔍 Google Search Grounding",
        value=True,
        help=(
            "Gemini automatically decides whether to search. "
            "It searches when the question needs up-to-date or real-world info. "
            "It won't search for simple questions it already knows the answer to."
        ),
    )

    use_code_exec = st.toggle(
        "💻 Code Execution",
        value=False,
        help=(
            "Gives Gemini a secure Python sandbox. "
            "It can write and run code to solve maths, analyse data, and generate charts (via Matplotlib). "
            "The sandbox has no internet access. Runs up to 30 seconds."
        ),
    )

    st.divider()

    # ── File upload ───────────────────────────
    st.markdown("**📎 Attach a File**")
    st.caption("Gemini can read and reason about the file in your conversation. Supported: images, PDFs, text files.")

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["jpg", "jpeg", "png", "webp", "gif", "pdf", "txt", "csv", "py", "js", "ts", "html", "md"],
        label_visibility="collapsed",
    )

    # Mime-type map for file types not auto-detected
    MIME_MAP = {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "webp": "image/webp",
        "gif":  "image/gif",
        "pdf":  "application/pdf",
        "txt":  "text/plain",
        "csv":  "text/csv",
        "py":   "text/x-python",
        "js":   "text/javascript",
        "ts":   "text/typescript",
        "html": "text/html",
        "md":   "text/markdown",
    }

    if uploaded_file:
        ext  = uploaded_file.name.split(".")[-1].lower()
        mime = MIME_MAP.get(ext, "application/octet-stream")

        # Preview the uploaded file in sidebar
        if mime.startswith("image/"):
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        else:
            st.success(f"📄 **{uploaded_file.name}** ({mime}) ready to send with your next message.")

        if st.button("🗑️ Remove file", use_container_width=True):
            uploaded_file = None
            st.rerun()

    st.divider()

    # ── Clear chat ────────────────────────────
    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
MAX_HISTORY = 30

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_mime(file) -> str:
    ext = file.name.split(".")[-1].lower()
    return MIME_MAP.get(ext, "application/octet-stream")


def render_part(part) -> str:
    """
    Render a single response Part from Gemini into Streamlit widgets.
    Returns a plain-text representation to save into chat history.
    """
    saved = ""

    # 1. Plain text / markdown
    if part.text:
        st.markdown(part.text)
        saved += part.text

    # 2. Code the model wrote and ran
    if part.executable_code:
        lang = str(part.executable_code.language).split(".")[-1].lower()
        if lang in ("language_unspecified", ""):
            lang = "python"
        code = part.executable_code.code
        st.markdown("**🖥️ Code Gemini ran:**")
        st.code(code, language=lang)
        saved += f"\n```{lang}\n{code}\n```\n"

    # 3. Output of that execution
    if part.code_execution_result:
        outcome = str(part.code_execution_result.outcome).lower()
        output  = part.code_execution_result.output or "(no output)"
        label   = "✅ Execution output:" if "ok" in outcome else "❌ Execution error:"
        st.markdown(f"**{label}**")
        st.code(output, language="text")
        saved += f"\n**{label}**\n```\n{output}\n```\n"

    # 4. Inline binary data (images from code execution, generated images, etc.)
    if part.inline_data:
        mime = part.inline_data.mime_type or ""
        data = part.inline_data.data  # bytes

        if mime.startswith("image/"):
            st.image(data, caption=f"Generated image ({mime})", use_container_width=True)
            saved += f"\n[Image returned: {mime}]\n"
        elif mime == "application/pdf":
            st.download_button(
                label="📄 Download PDF",
                data=data,
                file_name="gemini_output.pdf",
                mime="application/pdf",
            )
            saved += "\n[PDF returned]\n"
        else:
            ext = mime.split("/")[-1] if "/" in mime else "bin"
            st.download_button(
                label=f"⬇️ Download file (.{ext})",
                data=data,
                file_name=f"gemini_output.{ext}",
                mime=mime,
            )
            saved += f"\n[File returned: {mime}]\n"

    return saved


def render_sources(candidate) -> None:
    """Show collapsible web sources when search grounding was used."""
    try:
        meta   = candidate.grounding_metadata
        chunks = (meta.grounding_chunks or []) if meta else []
        if not chunks:
            return
        with st.expander("🔍 Web sources used by Gemini", expanded=False):
            for i, chunk in enumerate(chunks, 1):
                if hasattr(chunk, "web") and chunk.web:
                    title = chunk.web.title or chunk.web.uri
                    st.markdown(f"{i}. [{title}]({chunk.web.uri})")
    except Exception:
        pass


def build_tools() -> list | None:
    tools = []
    if use_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    if use_code_exec:
        tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
    return tools or None


def build_contents(user_text: str, file=None) -> list:
    """
    Build the full conversation contents list for the API call.
    Older history turns are plain text. The latest user message may include
    an attached file sent as inline bytes.
    """
    contents = []

    # Previous turns (text only — we don't re-send old files)
    history_turns = st.session_state.chat_history[-(MAX_HISTORY - 1):]
    for turn in history_turns:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=turn["content"])],
            )
        )

    # Current user turn (text + optional file)
    user_parts = []

    if file:
        file_bytes = file.read()
        mime       = get_mime(file)
        user_parts.append(
            types.Part.from_bytes(data=file_bytes, mime_type=mime)
        )

    user_parts.append(types.Part(text=user_text))

    contents.append(
        types.Content(role="user", parts=user_parts)
    )

    return contents


# ─────────────────────────────────────────────
# DISPLAY EXISTING HISTORY
# ─────────────────────────────────────────────
for msg in st.session_state.chat_history:
    role_label = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role_label):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
user_input = st.chat_input("Ask anything…")

if user_input:

    # ── Show & save the user message ──────────
    display_text = user_input
    if uploaded_file:
        display_text = f"📎 *[{uploaded_file.name}]*\n\n{user_input}"

    st.session_state.chat_history.append({"role": "user", "content": display_text})
    with st.chat_message("user"):
        st.markdown(display_text)

    # ── Build request ──────────────────────────
    tools  = build_tools()
    config = types.GenerateContentConfig(
        temperature=temperature,
        **({"tools": tools} if tools else {}),
    )
    contents = build_contents(user_input, file=uploaded_file)

    # ── Generate & display response ────────────
    with st.chat_message("assistant"):
        full_response = ""

        try:
            # Code execution produces multi-part responses that only arrive fully
            # assembled (not streamed), so we use non-streaming when it is on.
            # Text-only (with or without search grounding) uses streaming for
            # the real-time typing effect.

            if use_code_exec:
                # ── Non-streaming path ─────────────────
                response  = client.models.generate_content(
                    model=selected_model,
                    contents=contents,
                    config=config,
                )
                candidate = response.candidates[0]
                for part in candidate.content.parts:
                    full_response += render_part(part)
                render_sources(candidate)

            else:
                # ── Streaming path ─────────────────────
                placeholder   = st.empty()
                streamed_text = ""
                last_candidate = None

                for chunk in client.models.generate_content_stream(
                    model=selected_model,
                    contents=contents,
                    config=config,
                ):
                    candidate = chunk.candidates[0]
                    last_candidate = candidate
                    for part in candidate.content.parts:
                        if part.text:
                            streamed_text += part.text
                            placeholder.markdown(streamed_text + "▌")

                placeholder.markdown(streamed_text)
                full_response = streamed_text

                # Show grounding sources from the last chunk's metadata
                if use_search and last_candidate:
                    render_sources(last_candidate)

        except APIError as e:
            st.error(f"**Gemini API error:** {e.message}")
        except Exception as e:
            st.error(f"**Unexpected error:** {e}")

    # ── Save assistant reply ───────────────────
    if full_response.strip():
        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_response.strip()}
        )




# import streamlit as st
# from google import genai
# from google.genai import types
# from google.genai.errors import APIError
# import os
# import base64
# from dotenv import load_dotenv

# # ---------------------------------------------------------------------------
# # Environment & client setup
# # ---------------------------------------------------------------------------
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")

# if not api_key:
#     st.error("❌ GEMINI_API_KEY not found. Please set it in your .env file.")
#     st.stop()

# client = genai.Client(api_key=api_key)

# # ---------------------------------------------------------------------------
# # Page config
# # ---------------------------------------------------------------------------
# st.set_page_config(
#     page_title="Gemini Chat",
#     page_icon="💬",
#     layout="wide",
# )

# st.title("💬 Gemini Chat Interface")

# # ---------------------------------------------------------------------------
# # Sidebar — model options grouped by family
# # ---------------------------------------------------------------------------
# st.sidebar.header("⚙️ Settings")

# STABLE_MODELS = {
#     "Gemini 2.5 Pro (stable)":         "gemini-2.5-pro",
#     "Gemini 2.5 Flash (stable)":       "gemini-2.5-flash",
#     "Gemini 2.5 Flash-Lite (stable)":  "gemini-2.5-flash-lite",
# }

# PREVIEW_MODELS = {
#     "Gemini 3.1 Pro (preview)":        "gemini-3.1-pro-preview",
#     "Gemini 3 Flash (preview)":        "gemini-3-flash-preview",
#     "Gemini 3.1 Flash-Lite (preview)": "gemini-3.1-flash-lite-preview",
# }

# ALL_MODELS = {**STABLE_MODELS, **PREVIEW_MODELS}
# model_display_names = list(ALL_MODELS.keys())

# st.sidebar.markdown("**🧠 Model**")
# st.sidebar.caption("Stable = production-ready · Preview = cutting-edge")

# selected_display = st.sidebar.selectbox(
#     "Choose a model",
#     options=model_display_names,
#     index=1,          # default: Gemini 2.5 Flash
#     label_visibility="collapsed",
# )
# selected_model = ALL_MODELS[selected_display]

# # Show a badge next to the model name
# if selected_display in STABLE_MODELS:
#     st.sidebar.success(f"✅ Stable model selected")
# else:
#     st.sidebar.warning(f"⚠️ Preview model — may have rate limits")

# st.sidebar.divider()

# # Temperature
# temperature = st.sidebar.slider(
#     "🌡️ Temperature",
#     min_value=0.0,
#     max_value=2.0,
#     value=1.0,
#     step=0.1,
#     help="Higher = more creative. Lower = more focused.",
# )

# st.sidebar.divider()

# # Search grounding toggle
# use_search = st.sidebar.toggle(
#     "🔍 Google Search Grounding",
#     value=True,
#     help="Let Gemini search the web for up-to-date information.",
# )

# # Code execution toggle
# use_code_exec = st.sidebar.toggle(
#     "💻 Code Execution",
#     value=False,
#     help="Let Gemini write and run Python code to answer questions.",
# )

# st.sidebar.divider()

# # Clear chat
# if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
#     st.session_state.chat_history = []
#     st.rerun()

# st.sidebar.divider()
# st.sidebar.caption(f"Model ID: `{selected_model}`")

# # ---------------------------------------------------------------------------
# # Session state
# # ---------------------------------------------------------------------------
# MAX_HISTORY = 30

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # ---------------------------------------------------------------------------
# # Helper — render a single response part appropriately
# # ---------------------------------------------------------------------------
# def render_part(part):
#     """
#     Render one response Part from Gemini into Streamlit.
#     Handles: text, executable_code, code_execution_result, inline_data.
#     Returns the text content of the part (for saving to history), or "".
#     """
#     rendered_text = ""

#     # 1. Plain text / markdown
#     if part.text:
#         st.markdown(part.text)
#         rendered_text += part.text

#     # 2. Code Gemini wrote and executed
#     if part.executable_code:
#         lang = part.executable_code.language
#         if isinstance(lang, str):
#             lang_str = lang.lower()
#         else:
#             # It may be an enum — convert safely
#             lang_str = str(lang).split(".")[-1].lower()
#         code = part.executable_code.code
#         st.markdown("**🖥️ Code Gemini ran:**")
#         st.code(code, language=lang_str if lang_str != "language_unspecified" else "python")
#         rendered_text += f"\n```{lang_str}\n{code}\n```\n"

#     # 3. Output of that execution
#     if part.code_execution_result:
#         outcome = str(part.code_execution_result.outcome)
#         output  = part.code_execution_result.output or ""
#         label   = "✅ Output:" if "ok" in outcome.lower() else "❌ Execution error:"
#         st.markdown(f"**{label}**")
#         st.code(output, language="text")
#         rendered_text += f"\n```\n{output}\n```\n"

#     # 4. Inline binary data — images (charts, generated images, etc.)
#     if part.inline_data:
#         mime = part.inline_data.mime_type or ""
#         data = part.inline_data.data          # bytes

#         if mime.startswith("image/"):
#             st.image(data, caption=f"Generated image ({mime})", use_container_width=True)
#             rendered_text += f"\n[Image: {mime}]\n"

#         elif mime == "application/pdf":
#             st.download_button(
#                 label="📄 Download PDF",
#                 data=data,
#                 file_name="gemini_output.pdf",
#                 mime="application/pdf",
#             )
#             rendered_text += "\n[PDF attachment]\n"

#         else:
#             # Generic binary — offer a download
#             ext = mime.split("/")[-1] if "/" in mime else "bin"
#             st.download_button(
#                 label=f"⬇️ Download file (.{ext})",
#                 data=data,
#                 file_name=f"gemini_output.{ext}",
#                 mime=mime,
#             )
#             rendered_text += f"\n[File: {mime}]\n"

#     return rendered_text


# # ---------------------------------------------------------------------------
# # Helper — render search grounding sources
# # ---------------------------------------------------------------------------
# def render_sources(candidate):
#     """Show the web sources Gemini used when grounding is on."""
#     try:
#         meta = candidate.grounding_metadata
#         if not meta:
#             return
#         chunks = meta.grounding_chunks or []
#         if not chunks:
#             return
#         with st.expander("🔍 Web sources used", expanded=False):
#             for i, chunk in enumerate(chunks, 1):
#                 if hasattr(chunk, "web") and chunk.web:
#                     title = chunk.web.title or chunk.web.uri
#                     uri   = chunk.web.uri
#                     st.markdown(f"{i}. [{title}]({uri})")
#     except Exception:
#         pass   # sources display is optional


# # ---------------------------------------------------------------------------
# # Helper — build tool list from sidebar settings
# # ---------------------------------------------------------------------------
# def build_tools():
#     tools = []
#     if use_search:
#         tools.append(types.Tool(google_search=types.GoogleSearch()))
#     if use_code_exec:
#         tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
#     return tools or None


# # ---------------------------------------------------------------------------
# # Helper — build message history in the format the new SDK expects
# # ---------------------------------------------------------------------------
# def build_contents():
#     contents = []
#     for turn in st.session_state.chat_history[-MAX_HISTORY:]:
#         role = "user" if turn["role"] == "user" else "model"
#         contents.append(
#             types.Content(
#                 role=role,
#                 parts=[types.Part(text=turn["content"])],
#             )
#         )
#     return contents


# # ---------------------------------------------------------------------------
# # Display existing chat history
# # ---------------------------------------------------------------------------
# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # ---------------------------------------------------------------------------
# # Chat input & response
# # ---------------------------------------------------------------------------
# user_input = st.chat_input("Ask anything…")

# if user_input:
#     # Show & save the user message immediately
#     st.session_state.chat_history.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     # Build request config
#     tools = build_tools()
#     config_kwargs = dict(temperature=temperature)
#     if tools:
#         config_kwargs["tools"] = tools

#     config   = types.GenerateContentConfig(**config_kwargs)
#     contents = build_contents()

#     with st.chat_message("assistant"):
#         full_response = ""

#         try:
#             # -----------------------------------------------------------------
#             # Decide streaming vs non-streaming
#             #
#             # Streaming works well for pure-text answers.
#             # When code execution is ON the response contains non-text parts
#             # (executable_code, code_execution_result, inline_data) that only
#             # arrive in the final assembled response — so we use non-streaming
#             # in that case for simplicity and correctness.
#             # Search grounding works fine with streaming in the new SDK.
#             # -----------------------------------------------------------------

#             if use_code_exec:
#                 # --- Non-streaming path ---
#                 response = client.models.generate_content(
#                     model=selected_model,
#                     contents=contents,
#                     config=config,
#                 )
#                 candidate = response.candidates[0]
#                 for part in candidate.content.parts:
#                     full_response += render_part(part)

#                 render_sources(candidate)

#             else:
#                 # --- Streaming path (text + optional search grounding) ---
#                 placeholder = st.empty()
#                 streamed_text = ""

#                 for chunk in client.models.generate_content_stream(
#                     model=selected_model,
#                     contents=contents,
#                     config=config,
#                 ):
#                     # Chunks during streaming only carry text parts
#                     for part in chunk.candidates[0].content.parts:
#                         if part.text:
#                             streamed_text += part.text
#                             placeholder.markdown(streamed_text + "▌")

#                 placeholder.markdown(streamed_text)
#                 full_response = streamed_text

#                 # After streaming completes, show grounding sources if any
#                 # We need the final response for metadata — do a lightweight
#                 # non-streaming call only if search is on and we got content
#                 if use_search and full_response:
#                     try:
#                         final = client.models.generate_content(
#                             model=selected_model,
#                             contents=contents,
#                             config=config,
#                         )
#                         render_sources(final.candidates[0])
#                     except Exception:
#                         pass  # sources are optional, never break the chat

#         except APIError as e:
#             st.error(f"Gemini API error: {e.message}")
#         except Exception as e:
#             st.error(f"Unexpected error: {e}")
#             full_response = ""

#     # Save the assistant reply to history (text only — keeps history manageable)
#     if full_response:
#         st.session_state.chat_history.append(
#             {"role": "assistant", "content": full_response.strip()}
#         )

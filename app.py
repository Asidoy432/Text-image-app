import streamlit as st
import torch
from transformers import pipeline
from huggingface_hub import InferenceClient
from PIL import Image
import base64
import io
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AuraAI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(120, 80, 255, 0.15) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255, 100, 180, 0.10) 0%, transparent 60%),
        #0a0a0f !important;
}

[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.stDeployButton { display: none !important; }
footer { display: none !important; }

/* ── Main container ── */
[data-testid="stMainBlockContainer"] {
    max-width: 820px !important;
    margin: 0 auto !important;
    padding: 0 16px 120px !important;
}

/* ── App header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 28px 0 8px;
    margin-bottom: 4px;
}
.app-logo {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #a78bfa, #f472b6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.app-name {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    background: linear-gradient(90deg, #c4b5fd, #f9a8d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.app-tagline {
    font-size: 12px;
    color: #6b6880;
    margin-left: auto;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* ── Mode pills ── */
.mode-bar {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
}

/* ── Chat messages ── */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 8px;
}

.msg-row {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    animation: fadeUp 0.3s ease;
}
.msg-row.user { flex-direction: row-reverse; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
}
.avatar.ai {
    background: linear-gradient(135deg, #7c3aed, #db2777);
}
.avatar.user {
    background: #1e1b2e;
    border: 1px solid #2d2a40;
    font-size: 12px;
    color: #9d97c0;
}

.bubble {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 14.5px;
    line-height: 1.6;
    letter-spacing: 0.1px;
}
.bubble.ai {
    background: #13111f;
    border: 1px solid #1e1b30;
    border-top-left-radius: 4px;
    color: #ddd8f0;
}
.bubble.user {
    background: linear-gradient(135deg, #4c1d95, #831843);
    border: none;
    border-top-right-radius: 4px;
    color: #f3e8ff;
    text-align: right;
}
.bubble.image-bubble {
    background: #0f0d1a;
    border: 1px solid #1e1b30;
    padding: 10px;
    border-top-left-radius: 4px;
}
.bubble.image-bubble img {
    border-radius: 12px;
    max-width: 100%;
    display: block;
}
.img-caption {
    font-size: 11px;
    color: #5a5670;
    margin-top: 6px;
    font-style: italic;
}

/* ── Input bar ── */
.input-bar-container {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: linear-gradient(to top, #0a0a0f 70%, transparent);
    padding: 16px 0 24px;
    z-index: 999;
}
.input-bar-inner {
    max-width: 820px;
    margin: 0 auto;
    padding: 0 16px;
}

/* ── Streamlit widget overrides ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #13111f !important;
    border: 1px solid #2a2640 !important;
    border-radius: 14px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14.5px !important;
    padding: 14px 18px !important;
    caret-color: #a78bfa !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #6d28d9 !important;
    box-shadow: 0 0 0 3px rgba(109,40,217,0.15) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label { display: none !important; }

/* ── Buttons ── */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #6d28d9, #be185d) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    letter-spacing: 0.3px !important;
}
[data-testid="stButton"] button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] button:disabled {
    opacity: 0.3 !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* ── Radio (mode toggle) ── */
[data-testid="stRadio"] {
    background: #13111f !important;
    border: 1px solid #1e1b30 !important;
    border-radius: 12px !important;
    padding: 6px 10px !important;
    display: inline-flex !important;
}
[data-testid="stRadio"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    color: #9d97c0 !important;
}
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-size: 13px !important;
}

/* ── Success / error / warning ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13.5px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #a78bfa !important;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #1e1b30, transparent);
    margin: 16px 0;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px 40px;
    color: #3d3855;
}
.empty-icon {
    font-size: 48px;
    margin-bottom: 16px;
    opacity: 0.6;
}
.empty-title {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #4a4565;
    margin-bottom: 8px;
}
.empty-sub {
    font-size: 13.5px;
    color: #3a3555;
    line-height: 1.6;
}

/* ── Suggestion chips ── */
.chips-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 20px;
}
.chip {
    background: #13111f;
    border: 1px solid #1e1b30;
    border-radius: 20px;
    padding: 7px 14px;
    font-size: 12.5px;
    color: #6b6880;
    cursor: default;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2640; border-radius: 4px; }

/* ── Column gap fix ── */
[data-testid="stHorizontalBlock"] { gap: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "💬 Chat"

# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_text_model():
    return pipeline(
        "text-generation",
        model="gpt2",
        device=-1
    )

def generate_image_hf(prompt: str) -> Image.Image:
    client = InferenceClient(api_key=HF_TOKEN, provider="auto")
    return client.text_to_image(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-dev"
    )

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def chat_response(prompt: str, text_pipe) -> str:
    result = text_pipe(
        prompt,
        max_new_tokens=120,
        pad_token_id=50256,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
    )
    full = result[0]["generated_text"]
    # Return only the generated continuation, not the prompt
    continuation = full[len(prompt):].strip()
    return continuation if continuation else full

# ── Load model silently ───────────────────────────────────────────────────────
text_pipe = None
model_error = None
try:
    text_pipe = load_text_model()
except Exception as e:
    model_error = str(e)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-logo">✦</div>
    <div class="app-name">AuraAI</div>
    <div class="app-tagline">Powered by GPT-2 · FLUX · HuggingFace</div>
</div>
""", unsafe_allow_html=True)

if model_error:
    st.error(f"⚠️ Failed to load text model: {model_error}")

# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.radio(
    "mode",
    ["💬 Chat", "🖼️ Image"],
    horizontal=True,
    label_visibility="collapsed"
)
st.session_state.mode = mode

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Token warning ─────────────────────────────────────────────────────────────
if mode == "🖼️ Image" and not HF_TOKEN:
    st.warning("⚠️ Add `HF_TOKEN` in **Streamlit Secrets** (Settings → Secrets) to enable image generation.")

# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">✦</div>
        <div class="empty-title">What can I help you with?</div>
        <div class="empty-sub">
            Chat with AI or generate stunning images.<br>Switch modes using the toggle above.
        </div>
        <div class="chips-row">
            <span class="chip">✍️ Write a short story</span>
            <span class="chip">🤔 Explain a concept</span>
            <span class="chip">🖼️ A futuristic city at dusk</span>
            <span class="chip">🚀 Astronaut on Saturn</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        msg_type = msg.get("type", "text")

        if role == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="avatar user">you</div>
                <div class="bubble user">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if msg_type == "image":
                st.markdown(f"""
                <div class="msg-row ai">
                    <div class="avatar ai">✦</div>
                    <div class="bubble image-bubble">
                        <img src="data:image/png;base64,{content}" />
                        <div class="img-caption">Generated with FLUX.1-dev</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-row ai">
                    <div class="avatar ai">✦</div>
                    <div class="bubble ai">{content}</div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Clear button ──────────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Input area ────────────────────────────────────────────────────────────────
st.markdown('<div style="height:80px"></div>', unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

with col1:
    placeholder = (
        "Describe an image to generate..."
        if mode == "🖼️ Image"
        else "Ask me anything..."
    )
    user_input = st.text_input(
        "input",
        placeholder=placeholder,
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    send_label = "🎨 Paint" if mode == "🖼️ Image" else "➤ Send"
    send_disabled = (mode == "🖼️ Image" and not HF_TOKEN)
    send = st.button(send_label, disabled=send_disabled, use_container_width=True)

# ── Handle send ───────────────────────────────────────────────────────────────
if send and user_input.strip():
    user_msg = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": user_msg, "type": "text"})

    if mode == "🖼️ Image":
        with st.spinner("✦ Painting your vision..."):
            try:
                img = generate_image_hf(user_msg)
                b64 = pil_to_b64(img)
                st.session_state.messages.append({
                    "role": "ai",
                    "content": b64,
                    "type": "image"
                })
            except Exception as e:
                err = str(e)
                if "402" in err or "quota" in err.lower():
                    msg = "❌ HF free quota exceeded. Try again tomorrow."
                elif "401" in err or "unauthorized" in err.lower():
                    msg = "❌ Invalid HF_TOKEN. Check your Streamlit Secrets."
                elif "503" in err or "loading" in err.lower():
                    msg = "⏳ Model is warming up. Wait 30 seconds and try again."
                else:
                    msg = f"❌ {err}"
                st.session_state.messages.append({
                    "role": "ai", "content": msg, "type": "text"
                })
    else:
        if text_pipe:
            with st.spinner("✦ Thinking..."):
                try:
                    reply = chat_response(user_msg, text_pipe)
                    st.session_state.messages.append({
                        "role": "ai", "content": reply, "type": "text"
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "ai",
                        "content": f"❌ Error: {e}",
                        "type": "text"
                    })
        else:
            st.session_state.messages.append({
                "role": "ai",
                "content": "❌ Text model failed to load. Please refresh.",
                "type": "text"
            })

    st.rerun()

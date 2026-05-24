import streamlit as st
import requests
import torch
from transformers import pipeline

st.set_page_config(page_title="Multi-modal AI Pro", layout="wide")

HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# --- Text Generation (local, lightweight GPT-2) ---
@st.cache_resource
def load_text_model():
    return pipeline(
        "text-generation",
        model="gpt2",
        device=-1  # CPU only, fits in 1GB RAM
    )

# --- Image Generation via HF Inference API (no local loading) ---
def generate_image_api(prompt: str):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return response.content  # raw image bytes
    else:
        raise Exception(f"API Error {response.status_code}: {response.text}")

st.title("🎨 Multi-modal AI Generator")

with st.spinner("Loading text engine..."):
    try:
        text_pipe = load_text_model()
        st.success("✅ Engine Status: Online")
    except Exception as e:
        st.error(f"Boot Error: {e}")
        st.stop()

tab1, tab2 = st.tabs(["✍️ Text Generation", "🖼️ Image Generation"])

with tab1:
    prompt = st.text_input("Story Prompt:", "A lonely robot on Mars")
    if st.button("Generate Text"):
        with st.spinner("Writing..."):
            try:
                res = text_pipe(prompt, max_new_tokens=80, pad_token_id=50256)
                st.write(res[0]['generated_text'])
            except Exception as e:
                st.error(f"Text generation failed: {e}")

with tab2:
    if not HF_TOKEN:
        st.warning("⚠️ Add your HuggingFace token in Streamlit Secrets as `HF_TOKEN` to enable image generation.")
    img_prompt = st.text_input("Image Prompt:", "A robot sitting on a red rock, cinematic lighting")
    if st.button("Generate Image", disabled=not HF_TOKEN):
        with st.spinner("Painting... (may take 20–30 seconds on cold start)"):
            try:
                img_bytes = generate_image_api(img_prompt)
                st.image(img_bytes)
            except Exception as e:
                st.error(f"Image generation failed: {e}")

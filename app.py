import streamlit as st
import torch
from transformers import pipeline
from huggingface_hub import InferenceClient
from PIL import Image
import io

st.set_page_config(page_title="Multi-modal AI Pro", layout="wide")

HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# --- Text Generation (local GPT-2) ---
@st.cache_resource
def load_text_model():
    return pipeline(
        "text-generation",
        model="gpt2",
        device=-1
    )

# --- Image Generation via HF InferenceClient ---
def generate_image_hf(prompt: str) -> Image.Image:
    client = InferenceClient(
        model="stabilityai/stable-diffusion-2-1",
        token=HF_TOKEN
    )
    image = client.text_to_image(prompt)
    return image

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
        st.warning("⚠️ Add HF_TOKEN in Streamlit Secrets to enable image generation.")
    img_prompt = st.text_input("Image Prompt:", "A robot sitting on a red rock, cinematic lighting")
    if st.button("Generate Image", disabled=not HF_TOKEN):
        with st.spinner("Painting... (may take 20–30 seconds)"):
            try:
                image = generate_image_hf(img_prompt)
                st.image(image)
            except Exception as e:
                st.error(f"Image generation failed: {e}")

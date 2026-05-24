import streamlit as st
import torch
from transformers import pipeline
from huggingface_hub import InferenceClient
from PIL import Image

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

# --- Image Generation via HF Inference Providers ---
def generate_image_hf(prompt: str) -> Image.Image:
    client = InferenceClient(
        api_key=HF_TOKEN,
        provider="auto"
    )
    image = client.text_to_image(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-dev"
    )
    return image

# --- App UI ---
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
                res = text_pipe(
                    prompt,
                    max_new_tokens=80,
                    pad_token_id=50256,
                    do_sample=True,
                    temperature=0.9
                )
                st.write(res[0]['generated_text'])
            except Exception as e:
                st.error(f"Text generation failed: {e}")

with tab2:
    if not HF_TOKEN:
        st.warning("⚠️ Add HF_TOKEN in Streamlit Secrets to enable image generation.")

    img_prompt = st.text_input(
        "Image Prompt:",
        "A robot sitting on a red rock, cinematic lighting"
    )

    if st.button("Generate Image", disabled=not HF_TOKEN):
        with st.spinner("Painting... (may take 20–40 seconds)"):
            try:
                image = generate_image_hf(img_prompt)
                st.image(image, use_container_width=True)
            except Exception as e:
                error_msg = str(e)
                if "402" in error_msg or "quota" in error_msg.lower():
                    st.error("❌ HF free quota exceeded. Try again tomorrow or upgrade your HF account.")
                elif "401" in error_msg or "unauthorized" in error_msg.lower():
                    st.error("❌ Invalid HF_TOKEN. Check your Streamlit Secrets.")
                elif "503" in error_msg or "loading" in error_msg.lower():
                    st.error("⏳ Model is loading on HF servers. Wait 30 seconds and try again.")
                else:
                    st.error(f"❌ Image generation failed: {e}")

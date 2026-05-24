import streamlit as st
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="Multi-modal AI Pro", layout="wide")

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = "Asidoy432/Text-image"

    # Explicitly load Text Pipeline using the subfolder path
    text_pipe = pipeline(
        "text-generation",
        model=f"{repo_id}",
        model_kwargs={"subfolder": "gpt2"},
        device=0 if device == "cuda" else -1
    )

    # For Stable Diffusion, we point directly to the repo. 
    # If components are in a subfolder, we use from_pretrained on the repo with the subfolder
    pipe = StableDiffusionPipeline.from_pretrained(
        repo_id,
        subfolder="stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    )

    pipe.to(device)
    return text_pipe, pipe, device

st.title("🎨 Multi-modal AI Generator")

with st.spinner("Initializing AI Engine... (2-5 mins on first run)"):
    try:
        text_pipe, pipe, device = load_models()
        st.success("Engine Status: Online")
    except Exception as e:
        st.error(f"Boot Error: {e}")
        st.stop()

tab1, tab2 = st.tabs(["✍️ Text Generation", "🖼️ Image Generation"])

with tab1:
    prompt = st.text_input("Story Prompt:", "A lonely robot on Mars")
    if st.button("Generate Text"):
        with st.spinner("Writing..."):
            res = text_pipe(prompt, max_new_tokens=50, pad_token_id=50256)
            st.write(res[0]['generated_text'])

with tab2:
    img_prompt = st.text_input("Image Prompt:", "A robot sitting on a red rock, cinematic lighting")
    if st.button("Generate Image"):
        with st.spinner("Painting..."):
            image = pipe(img_prompt).images[0]
            st.image(image)

# Sync: Sun May 24 14:46:45 2026
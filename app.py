import streamlit as st
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image

st.set_page_config(page_title="Multi-modal AI Pro", layout="wide")

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = "Asidoy432/Text-image"

    # Load Text Pipeline from HF - Fixed to include subfolder
    text_pipe = pipeline(
        "text-generation", 
        model=repo_id, 
        model_kwargs={"subfolder": "gpt2"},
        device=0 if device == "cuda" else -1
    )

    # Load Image Pipeline from HF
    pipe = StableDiffusionPipeline.from_pretrained(
        repo_id,
        subfolder="stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe.to(device)
    return text_pipe, pipe, device

st.title("🎨 Multi-modal AI Generator")

with st.spinner("Loading models from Hugging Face... This takes a few minutes on first run."):
    try:
        text_pipe, pipe, device = load_models()
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

tab1, tab2 = st.tabs(["✍️ Text", "🖼️ Image"])
with tab1:
    st.header("Text-to-Text")
    input_text = st.text_input("Story prompt:", "Once upon a time")
    if st.button("Generate"):
        with st.spinner("Generating text..."):
            result = text_pipe(input_text, max_new_tokens=50, pad_token_id=50256)
            st.write(result[0]['generated_text'])
with tab2:
    st.header("Text-to-Image")
    input_img = st.text_input("Image prompt:", "A futuristic forest")
    if st.button("Paint"):
        with st.spinner("Generating image..."):
            image = pipe(input_img).images[0]
            st.image(image)

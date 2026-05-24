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

    # Load Text Pipeline - Ensuring subfolder is handled for tokenizer and model
    text_pipe = pipeline(
        "text-generation",
        model=repo_id,
        model_kwargs={"subfolder": "gpt2"},
        device=0 if device == "cuda" else -1
    )

    # Load Image Pipeline - Explicitly setting subfolder
    pipe = StableDiffusionPipeline.from_pretrained(
        repo_id,
        subfolder="stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    )
    pipe.to(device)
    return text_pipe, pipe, device

st.title("🎨 Multi-modal AI Generator")

with st.spinner("Loading models from Hugging Face... This takes a few minutes on first run."):
    try:
        text_pipe, pipe, device = load_models()
        st.success(f"Models loaded successfully on {device.upper()}!")
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

tab1, tab2 = st.tabs(["✍️ Text Generation", "🖼️ Image Generation"])

with tab1:
    st.header("Text-to-Text")
    input_text = st.text_input("Enter a story prompt:", "The scientist discovered a mysterious portal")
    if st.button("Generate Text"):
        with st.spinner("Generating..."):
            result = text_pipe(input_text, max_new_tokens=50, pad_token_id=50256)
            st.write(result[0]['generated_text'])

with tab2:
    st.header("Text-to-Image")
    input_img = st.text_input("Describe the image:", "A futuristic laboratory with glowing portals")
    if st.button("Generate Image"):
        with st.spinner("Painting (this can take 2-5 mins on CPU)..."):
            try:
                image = pipe(input_img).images[0]
                st.image(image, caption=input_img)
            except Exception as e:
                st.error(f"Inference Error: {e}")

# Updated deployment configuration
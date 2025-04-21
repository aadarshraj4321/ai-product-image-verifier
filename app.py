import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from utils.image_quality import detect_blur, check_brightness, check_contrast, check_resolution

# Set device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to get CLIP similarity score
def get_clip_similarity(image, text):
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image  # image-text similarity
    probs = logits_per_image.softmax(dim=1)
    return probs[0][0].item()

# Function to get image caption using BLIP
def get_caption(pil_image):
    inputs = caption_processor(images=pil_image, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)

# UI Setup
st.set_page_config(page_title="Product Image Verifier", layout="centered")

# Title and Intro
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1>AI Product Image Verifier</h1>
    <p>Ensure your product images match the description and meet quality standards with AI.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""---""")

# User input fields
uploaded_img = st.file_uploader("Upload Product Image", type=['jpg', 'jpeg', 'png'])
description = st.text_area("Enter Product Description")

if uploaded_img and description:
    image = Image.open(uploaded_img)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Image Check"):
        with st.spinner('Analyzing with AI...'):
            # CLIP similarity check
            st.subheader("Image-Text Relevance (CLIP Model)")
            sim = get_clip_similarity(image, description)
            st.progress(sim)
            st.metric(label="Similarity Score", value=f"{sim:.2f}")

            # Get caption using BLIP
            caption = get_caption(image)
            st.write(f"AI-generated Caption: `{caption}`")

            # Keyword check for image-caption and description
            if any(word.lower() in caption.lower() for word in description.split()):
                st.success("Description matches image content")
            else:
                st.warning("Description may not match image content")

            # Image Quality Metrics
            st.subheader("Image Quality Analysis")
            blur = detect_blur(image)
            brightness = check_brightness(image)
            contrast = check_contrast(image)
            width, height = check_resolution(image)

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Blur Score: `{blur:.2f}` (higher is better)")
                st.write(f"Brightness: `{brightness:.2f}`")
            with col2:
                st.write(f"Contrast: `{contrast:.2f}`")
                st.write(f"Resolution: `{width} x {height}`")

            st.subheader("AI Verdict")
            with st.expander("See Evaluation Logic"):
                st.code("""
if sim > 0.35 and blur > 100 and brightness > 60 and contrast > 40 and width >= 600:
    st.success("Image is Good to Upload")
else:
    st.warning("Image has some issues")
""")

            if sim > 0.35 and blur > 100 and brightness > 60 and contrast > 40 and width >= 600:
                st.success("Image is Good to Upload")
            else:
                st.warning("Image has some issues")

        if st.button("Final Submit & Save Report"):
            st.balloons()
            st.success("Report Submitted!")

else:
    st.info("Please upload an image and write a description to start.")

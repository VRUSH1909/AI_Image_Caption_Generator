import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# ======================= PAGE CONFIG =======================
st.set_page_config(page_title="AI Image Caption Generator", layout="centered")

# ======================= CUSTOM STYLES =======================
def apply_theme(theme):
    if theme == "Dark":
        st.markdown("""
        <style>
            body, .stApp {
                background-color: #0e1117;
                color: #ffffff;
            }
            .block-container {
                padding-top: 2rem;
            }
            .stButton > button {
                background-color: #6c63ff;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
            }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                color: #f9f9f9;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            body, .stApp {
                background-color: #ffffff;
                color: #000000;
            }
            .block-container {
                padding-top: 2rem;
            }
            .stButton > button {
                background-color: #6c63ff;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
            }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                color: #444;
            }
        </style>
        """, unsafe_allow_html=True)

# ======================= LOAD MODELS =======================
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_vit_gpt2_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, feature_extractor, tokenizer

blip_processor, blip_model = load_blip_model()
vit_gpt2_model, vit_feature_extractor, vit_tokenizer = load_vit_gpt2_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_gpt2_model.to(device)

# ======================= CAPTION GENERATION =======================
def generate_caption_blip(image):
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def generate_caption_vit_gpt2(image):
    pixel_values = vit_feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = vit_gpt2_model.generate(pixel_values, max_length=16, num_beams=4)
    return vit_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

# ======================= STREAMLIT UI =======================
with st.sidebar:
    st.title("🎨 UI Settings")
    theme = st.radio("Choose Theme", ["Dark", "Light"], index=0)  # Dark as default
    st.markdown("Toggle between light and dark themes.")

apply_theme(theme)

st.title("🧠 AI Image Captioning App")
st.markdown("Upload an image and generate creative captions using **two different AI models**. ✨")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    if st.button("⚡ Generate Captions"):
        with st.spinner("Thinking... 🤖"):
            caption_blip = generate_caption_blip(image)
            caption_vit = generate_caption_vit_gpt2(image)

        st.success("✨ Captions generated successfully!")

        st.markdown("### 📋 Captions from AI Models")
        st.markdown(f"**🔹 BLIP Model:** _{caption_blip}_")
        st.markdown(f"**🔹 ViT-GPT2 Model:** _{caption_vit}_")

        st.markdown("---")
        st.info("Try uploading another image or switch the theme using the sidebar.")

# ======================= FOOTER =======================
st.markdown("""
<hr style="border: 1px solid #444;">
<center><small>🤖 Captioned by Transformers. Styled by Streamlit. | AI-powered Image Captioning</small></center>
""", unsafe_allow_html=True)

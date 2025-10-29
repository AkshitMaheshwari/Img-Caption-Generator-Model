import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle

# =============== Configuration ===============
MODEL_PATH = "model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
FEATURE_EXTRACTOR_PATH = "feature_extractor.keras"
MAX_LENGTH = 34
IMG_SIZE = 224

# =============== Load Resources ===============
@st.cache_resource
def load_resources():
    caption_model = load_model(MODEL_PATH)
    feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return caption_model, feature_extractor, tokenizer

caption_model, feature_extractor, tokenizer = load_resources()

# =============== Caption Generation Function ===============
def generate_caption(image, caption_model, feature_extractor, tokenizer, max_length=MAX_LENGTH, img_size=IMG_SIZE):
    # Preprocess the image
    img = load_img(image, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Extract features
    image_features = feature_extractor.predict(img, verbose=0)

    # Generate caption
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption

# =============== Display Image with Caption ===============
def display_image_with_caption(image, caption, img_size=IMG_SIZE):
    img = load_img(image, target_size=(img_size, img_size))
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption, fontsize=16, color='blue')
    st.pyplot(plt)

# =============== Streamlit App UI ===============
st.set_page_config(page_title="🖼️ Image Caption Generator", page_icon="🧠", layout="centered")

st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image, and the model will generate a caption for it.")

uploaded_file = st.file_uploader("📤 Upload or drag an image here", type=["jpg", "jpeg", "png","webp"])

if uploaded_file is not None:
    if st.button("✨ Generate Caption"):
        with st.spinner("Generating caption... ⏳"):
            caption = generate_caption(uploaded_file, caption_model, feature_extractor, tokenizer)
        st.success("✅ Caption Generated:")
        display_image_with_caption(uploaded_file, caption)
else:
    st.info("Please upload an image to generate a caption.")

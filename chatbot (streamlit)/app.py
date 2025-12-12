import streamlit as st
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(page_title="Tomato Disease Detector", page_icon="🍅", layout="wide")

# Show loading
st.title("🍅 Tomato Leaf Disease Detector")
loading_msg = st.empty()
loading_msg.info("⏳ Loading model... please wait")

# Import TF after page config
import tensorflow as tf

# Get directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Disease info
INFO = {
    "Tomato___Bacterial_spot": ("Bacterial Spot", "Remove infected leaves, apply copper spray"),
    "Tomato___Early_blight": ("Early Blight", "Apply fungicides, remove lower leaves"),
    "Tomato___Late_blight": ("Late Blight", "Apply fungicides immediately, destroy infected plants"),
    "Tomato___Leaf_Mold": ("Leaf Mold", "Reduce humidity, improve ventilation"),
    "Tomato___Septoria_leaf_spot": ("Septoria Leaf Spot", "Remove infected leaves, apply fungicides"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Spider Mites", "Spray water, use neem oil"),
    "Tomato___Target_Spot": ("Target Spot", "Apply fungicides, ensure proper spacing"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Yellow Leaf Curl Virus", "Control whiteflies, remove infected plants"),
    "Tomato___Tomato_mosaic_virus": ("Mosaic Virus", "Remove infected plants, sanitize tools"),
    "Tomato___healthy": ("Healthy Plant", "No treatment needed - keep up good care!")
}

# Default classes
DEFAULT_CLASSES = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

@st.cache_resource
def load_model_safe():
    model = None
    config = None
    error = None
    
    # Load config
    config_path = os.path.join(BASE_DIR, 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {'class_names': DEFAULT_CLASSES, 'img_size': 224, 'confidence_threshold': 0.6}
    
    # Try loading model
    model_files = ['tomato_model.h5', 'tomato_disease_model.h5', 'tomato_model.keras', 'tomato_disease_model.keras']
    
    for mf in model_files:
        model_path = os.path.join(BASE_DIR, mf)
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                break
            except Exception as e:
                error = str(e)
    
    return model, config, error

# Load model
model, config, error = load_model_safe()
loading_msg.empty()

# Check if loaded
if model is None:
    st.error("❌ Could not load model")
    st.error(f"Error: {error}")
    st.stop()

# Get settings
CLASSES = config.get('class_names', DEFAULT_CLASSES)
SIZE = config.get('img_size', 224)
THRESH = config.get('confidence_threshold', 0.6)

st.success(f"✅ Model loaded successfully!")

def predict(img):
    img = img.resize((SIZE, SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return model.predict(arr, verbose=0)[0]

# Main UI
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    file = st.file_uploader("Choose a tomato leaf image", type=['jpg', 'jpeg', 'png'])
    
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # ANALYZE BUTTON
        analyze_btn = st.button("🔍 Analyze Disease", type="primary", use_container_width=True)

with col2:
    st.subheader("🔬 Results")
    
    if file:
        # Only analyze when button is clicked
        if 'analyze_btn' in dir() and analyze_btn:
            with st.spinner("🔍 Analyzing leaf..."):
                preds = predict(img)
            
            idx = int(np.argmax(preds))
            conf = float(preds[idx])
            cls = CLASSES[idx]
            name, treat = INFO.get(cls, (cls.replace("Tomato___", ""), "Consult an expert"))
            
            if conf < THRESH:
                st.warning(f"⚠️ **Low Confidence: {conf:.0%}**")
                st.info("This might not be a tomato leaf or image quality is poor.")
                
                st.markdown("**Best guesses:**")
                for i in np.argsort(preds)[-3:][::-1]:
                    n = INFO.get(CLASSES[i], (CLASSES[i],))[0]
                    st.write(f"• {n}: {preds[i]:.1%}")
            else:
                if "healthy" in cls.lower():
                    st.success(f"✅ **{name}**")
                    st.balloons()
                else:
                    st.error(f"🦠 **Disease: {name}**")
                
                st.metric("Confidence", f"{conf:.0%}")
                st.progress(conf)
                
                st.markdown("---")
                st.info(f"💊 **Treatment:** {treat}")
            
            with st.expander("📊 All Predictions"):
                for i in np.argsort(preds)[::-1]:
                    if preds[i] > 0.01:
                        n = INFO.get(CLASSES[i], (CLASSES[i],))[0]
                        st.write(f"{n}: {preds[i]:.1%}")
        else:
            st.info("👈 Upload an image and click **Analyze Disease** button")
    else:
        st.info("👆 Upload a tomato leaf image to get started")
        st.markdown("""
        **Tips for best results:**
        - Use a clear, focused image
        - Ensure good lighting
        - Show the leaf clearly
        """)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write(f"**Classes:** {len(CLASSES)}")
    st.write(f"**Threshold:** {THRESH:.0%}")
    
    st.markdown("---")
    st.header("🦠 Detectable Diseases")
    for c in CLASSES:
        n = INFO.get(c, (c,))[0]
        e = "✅" if "healthy" in c.lower() else "🔴"
        st.write(f"{e} {n}")
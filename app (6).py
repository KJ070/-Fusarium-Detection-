import streamlit as st
import numpy as np
import imageio
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define severity level mapping
severity_levels = {
    0: "Highly Resistant (HR)",
    1: "Resistant (R)",
    2: "Moderate (M)",  # Adjust as needed
    3: "Susceptible (S)",
    4: "Highly Susceptible (HS)"
}

# Streamlit UI
st.set_page_config(page_title="DLgram Deployment", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #00FF00;'>DLgram Deployment</h1>
    
    <hr>
    """, unsafe_allow_html=True)

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = imageio.imread(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True, width=300)  # Adjusted size and updated parameter
    
    # Preprocess the image
    img = img_to_array(load_img(uploaded_file, target_size=(224, 224)))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalization
    
    # Load Model
    @st.cache_resource
    def load_model():
        model = tf.keras.models.load_model("model.keras")
        return model
    
    model = load_model()
    
    # Make Prediction
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)  # Get the class with highest probability
    
    # Display results
    st.markdown("<h3 style='text-align: center;'>Prediction Results</h3>", unsafe_allow_html=True)
    for index, score in enumerate(prediction[0]):
        st.write(f"<b>{severity_levels.get(index, 'Unknown')}</b>: {score:.4f}", unsafe_allow_html=True)
    
    # Highlight the predicted class
    st.success(f"Final Prediction: **{severity_levels.get(predicted_label, 'Unknown')}**")

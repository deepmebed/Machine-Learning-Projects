import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('skin_cancer_model.h5')

# Function to predict the image
def predict_image(img, model):
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Malignant"
    else:
        return "Benign"

# Streamlit interface
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    .title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2e8b57; /* Changed heading color */
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2e8b57;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .file-uploader {
        margin-bottom: 20px;
    }
    .image {
        margin-bottom: 20px;
    }
    .result {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 40px;
        color: #1e90ff; /* Changed result color */
        text-align: center;
        margin-top: 20px;
        font-weight: bold;
    }
    /* Change the color of the upload button */
    .file_input_button::before {
        background-color: #2e8b57; /* Changed button color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("EpiDerm: Skin Cancer Detection WebApp")
st.markdown(
    """
    <p>Welcome to EpiDerm, where cutting-edge technology meets dermatological expertise to provide precision skin analysis. Our platform integrates advanced AI algorithms with comprehensive dermatological data to revolutionize skin health assessment. Simply upload an image of your skin lesion, and our AI model will provide an instant diagnosis. EpiDerm is your trusted partner in skin cancer detection and prevention.</p>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key='fileUploader')

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict and display the result
    result = predict_image(img, model)
    st.markdown(f"<div class='result'>{result}</div>", unsafe_allow_html=True)

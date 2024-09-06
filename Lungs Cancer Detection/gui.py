import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
from time import sleep

# Load the trained models
pca = joblib.load('pca_model.pkl')
svm = joblib.load('svm_model.pkl')

# Categories
categories = ['Normal case', 'Malignant case', 'Bengin case']

# Custom CSS for styling
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            color: #003366;
            text-align: center;
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .description {
            font-size: 18px;
            color: #003366;
            text-align: center;
            padding-bottom: 20px;
        }
        .upload-box {
            padding: 20px;
            border: 2px dashed #003366;
            border-radius: 10px;
            text-align: center;
        }
        .upload-text {
            font-size: 20px;
            color: #003366;
            padding-top: 10px;
        }
        .prediction-result {
            font-size: 24px;
            color: #003366;
            text-align: center;
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .thank-you {
            font-size: 18px;
            color: #003366;
            text-align: center;
            padding-top: 20px;
            padding-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown("<h1 class='main-title'>Lung Cancer Detection</h1>", unsafe_allow_html=True)
st.markdown("<div class='description'>Upload a CT scan image, and our AI model will predict if it's a normal case, malignant case, or benign case. This tool aids in the early detection and treatment of lung cancer.</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = np.array(image.convert('L'))  # Convert to grayscale
    img = cv2.resize(img, (128, 128))   # Resize to the same size as training images
    img_flatten = img.flatten().reshape(1, -1)
    
    # Apply PCA
    img_pca = pca.transform(img_flatten)
    
    # Predict using SVM
    with st.spinner(text='Predicting...'):
        sleep(2)  # Simulate prediction time
        prediction = svm.predict(img_pca)

    # Display the result with animation
    st.markdown("<div class='prediction-result'>Prediction: <span style='color:#ff6600;'>{}</span></div>".format(categories[prediction[0]]), unsafe_allow_html=True)

# Final Description
st.markdown("""
    <div class='description'>
        Our AI-powered lung cancer detection system analyzes lung CT scan images, the model assists healthcare professionals in making informed decisions, potentially saving lives through early detection.
    </div>
    <div class='thank-you'>
        Thank you for visiting our Lung Cancer Detection tool. We hope you found it helpful. Stay healthy and take care!
    </div>
""", unsafe_allow_html=True)

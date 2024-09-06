# -------------------------------------------- Importing Libraries --------------------------------------------

import streamlit as st
import joblib
from PIL import Image
import numpy as np
from skimage import color, transform, feature

# Assuming 'set_background' is defined in 'util.py'
# from util import set_background

# Set the background image
# set_background('./bgs/bg5.png')

# -------------------------------------------- Main Code --------------------------------------------

# -------------------------------------------- Streamlit App --------------------------------------------

st.set_page_config(
    page_title="Mango Types Classifier",
    page_icon="🥭",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -------------------------------------------- Add CSS Styling --------------------------------------------

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
        color: #2e8b57;
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
        font-size: 20px;
        color: #2e8b57;
        text-align: center;
        margin-top: 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="title">Mango Types Classifier</h1>', unsafe_allow_html=True)

st.markdown('<h3 class="header">Please upload a mango image to classify the type of mango</h3>', unsafe_allow_html=True)

#-------------------------------------------- Load the Classifier and Take Image input  --------------------------------------------

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'], key='file_uploader')

model = joblib.load('./model/mango_classifier.pkl')

#-------------------------------------------- Feature Extraction and Classification  --------------------------------------------

def extract_features(image):
    # Convert to grayscale
    image_gray = color.rgb2gray(image)
    # Resize image
    image_resized = transform.resize(image_gray, (64, 64), anti_aliasing=True)
    # Extract HOG features
    hog_features = feature.hog(image_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return hog_features

def classify(image, model):
    # Preprocess the image
    image_np = np.array(image)
    features = extract_features(image_np)
    # Predict the type of mango using the classifier
    prediction = model.predict([features])
    # Get the class name using the prediction
    # class_name = class_names[prediction[0]]
    # Here you might want to calculate the confidence score or probability
    # conf_score = np.max(model.predict_proba([features])) # Example to get the confidence score
    return prediction[0]


# -------------------------------------------- Display the Image and Classification Result --------------------------------------------

# If a file is uploaded, display the image and classify it
if file is not None:
    # Open the image
    image = Image.open(file).convert('RGB')
    # Display the image
    st.image(image, use_column_width=True, caption="Uploaded Mango Image", output_format="auto")

    # Classify the image
    class_name = classify(image, model)

    # Write the classification result and confidence score
    st.markdown(f'<div class="result">Mango Type: {class_name}</div>', unsafe_allow_html=True)

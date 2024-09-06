import streamlit as st
from PIL import Image
import joblib
import cv2
import numpy as np


def image_to_vector(image, size):
    return cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC).flatten()

def classify_single_image(image, model, size):
    image_resized = cv2.resize(image, size)
    if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    image_features = image_to_vector(image_resized, size).reshape(1, -1)

    prediction = model.predict(image_features)

    

    label = "Hemorrhage" if prediction == 1 else "No Hemorrhage"
    
    return label

model = joblib.load('trained_model.joblib')

with open('styles.html', 'r') as file:
    css = file.read()
st.markdown(css, unsafe_allow_html=True)



# Load and apply the background HTML and CSS
st.markdown(
    """
    <style>
    @import url('background.css');
    </style>
    """,
    unsafe_allow_html=True
)

with open('background.html', 'r') as file:
    background_html = file.read()
st.markdown(background_html, unsafe_allow_html=True)

# Set the title of the Streamlit app
st.markdown('<div class="header">Brain Hemorrhage Detector</div>', unsafe_allow_html=True)

# Introductory text
st.markdown(
    """
    <div class="intro">
        Welcome to the Brain Hemorrhage Detector. This tool allows you to upload a brain CT scan image,
        and our machine learning model will analyze the image to detect whether there is a hemorrhage.
        Simply upload an image, and click on 'Predict' to see the result.
    </div>
    """,
    unsafe_allow_html=True
)

# File uploader for image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image file
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True, output_format='PNG', clamp=True, channels="RGB")

    # Define the target size for image resizing
    target_size = (256, 256)  # Adjust to the size used during training

    # Predict button
    if st.button('Predict', key='predict', help='Click to predict whether the image shows a brain hemorrhage.', type='primary'):
        # Convert the uploaded image to numpy array
        img_array = np.array(image)

        # Ensure the image has three channels (RGB) by converting if necessary
        if len(img_array.shape) == 2:  # if grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # if RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Convert image to grayscale if needed
        if img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Run the classification model
        prediction = classify_single_image(img_array, model, target_size)

        # Display the prediction result with color formatting
        result_class = 'no-hemorrhage' if prediction == 'No Hemorrhage' else 'hemorrhage'
        result_text = f"Prediction: {prediction}"
        st.markdown(f'<div class="result-box {result_class}">{result_text}</div>', unsafe_allow_html=True)


        # Thank you message
        st.markdown(
            """
            <div class="thank-you">
                Thank you for using the Brain Hemorrhage Detection app. We hope you found it helpful!
            </div>
            """,
            unsafe_allow_html=True
        )

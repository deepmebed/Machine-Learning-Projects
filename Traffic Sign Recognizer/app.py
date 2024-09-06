import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Traffic SignSpotter", layout="wide")

# Load the saved components
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
svm_model = joblib.load('svm_model.pkl')

# Label mapping
label_mapping = {
    'airport': 0,
    'Animal_Crossing_Sign': 1,
    'Bridge Ahead': 2,
    'Bus_Stop': 3,
    'Cross Roads': 4,
    'Danger_Ahead': 5,
    'Dense_Ahead': 6,
    'Dinning_Place': 7,
    'DownHeal_Step': 8,
    'Give Way': 9,
    'Go_Straight_ahead': 10,
    'Hospital': 11,
    'Land_Sliding': 12,
    'Left bend': 13,
    'MotorWay_Start': 14,
    'No entry for Bikes': 15,
    'No entry for car': 16,
    'No entry for Cycles': 17,
    'No entry for Goods vehicle': 18,
    'No entry for hand crafts': 19,
    'no entry for vehical more than 16.6 feet': 20,
    'No Entry Vehicle weight 70 ton': 21,
    'No Horns': 22,
    'No left turn': 23,
    'No Mobile Allowed': 24,
    'No Overtaking': 25,
    'No Parking': 26,
    'No right turn': 27,
    'No U-Turn': 28,
    'no walking for pedistrians': 29,
    'No_Entry for aniamal vehicle': 30,
    'One Way ROad': 31,
    'parking on left': 32,
    'Pedestrians': 33,
    'petrol pump 3': 34,
    'Railway Crossing': 35,
    'Right bend': 36,
    'right turn': 37,
    'road crossing': 38,
    'Road Divides': 39,
    'Roundabout Ahead': 40,
    'Sharp Right Turn': 41,
    'Slow': 42,
    'Speed Breaker Ahead': 43,
    'Speed Limit (20 kmph)': 44,
    'Speed Limit (25 kmph)': 45,
    'Speed Limit (30 kmph)': 46,
    'Speed Limit (40 kmph)': 47,
    'Speed Limit (45 kmph)': 48,
    'Speed Limit (50 kmph)': 49,
    'Speed Limit (60 kmph)': 50,
    'Speed Limit (65 kmph)': 51,
    'Speed Limit (70 kmph)': 52,
    'Speed Limit (80 kmph)': 53,
    'Steep Descent': 54,
    'Stop': 55,
    'two way traffic 2': 56,
    'uphill steep': 57,
    'U-Turn': 58,
    'Zigzag Road Ahead': 59
}

# Reverse label mapping
label_mapping_rev = {v: k for k, v in label_mapping.items()}

# Preprocess the uploaded image
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    img = cv2.resize(img, (32, 32))  # Resize image to 32x32 pixels
    return img

# Extract HOG features from the image
def extract_hog_features(images):
    hog_features = []
    for img in images:
        features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), block_norm='L2-Hys',
                          visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

# Predict the label of the image
def predict_image(img_path):
    try:
        img = preprocess_image(img_path)
    except FileNotFoundError as e:
        st.error(e)
        return None

    img = np.array([img])  # Convert to a batch of 1 image
    
    hog_features = extract_hog_features(img)
    hog_features = scaler.transform(hog_features)
    hog_features_pca = pca.transform(hog_features)
    
    prediction = svm_model.predict(hog_features_pca)
    
    predicted_label = label_mapping_rev.get(prediction[0], "Unknown")
    
    # Display the image with the predicted label
    st.image(Image.open(img_path), caption=f'Predicted Label: {predicted_label}', use_column_width=True)
    
    return predicted_label

# Streamlit app layout
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        color: white;
    }
    .title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .subheader {
        text-align: center;
        font-size: 1.5em;
        margin-bottom: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Traffic SignSpotter</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload an image to classify</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    predicted_label_name = predict_image("uploaded_image.jpg")
    if predicted_label_name:
        st.markdown(f'<div style="text-align:center; font-size:3.5em; font-weight:bold; color:#FF0000;">{predicted_label_name}</div>', unsafe_allow_html=True)
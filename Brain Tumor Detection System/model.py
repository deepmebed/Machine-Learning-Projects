import cv2
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMAGE_SIZE = 300
base_path = r'C:\Users\samsaam\OneDrive\Documents\brain t\Brain-Tumor-Detection-main'
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    feature_vector = img.flatten()
    return feature_vector

def preprocess_image_from_array(image_array):
    img = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    feature_vector = img.flatten()
    return feature_vector

def load_data():
    x_data = []
    y_data = []

    for i in class_names:
        folderPath = os.path.join(base_path, 'Training', i)
        for j in tqdm(os.listdir(folderPath), ncols=70):
            img_path = os.path.join(folderPath, j)
            feature_vector = preprocess_image(img_path)
            x_data.append(feature_vector)
            y_data.append(i)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data

def train_model():
    x_data, y_data = load_data()
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, random_state=47, test_size=0.20)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, random_state=47, test_size=0.50)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=300)
    x_train_pca = pca.fit_transform(x_train)
    x_val_pca = pca.transform(x_val)
    x_test_pca = pca.transform(x_test)

    # Initialize SVM classifier with specific hyperparameters
    svm_classifier = SVC(C=10, kernel='rbf', gamma='scale')

    # Train the SVM classifier
    svm_classifier.fit(x_train_pca, y_train)

    # Make predictions
    train_predictions = svm_classifier.predict(x_train_pca)
    val_predictions = svm_classifier.predict(x_val_pca)
    test_predictions = svm_classifier.predict(x_test_pca)

    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, train_predictions)
    val_accuracy = accuracy_score(y_val, val_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Print accuracies
    # print("Training Accuracy (SVM):", train_accuracy)
    # print("Validation Accuracy (SVM):", val_accuracy)
    print("Testing Accuracy (SVM):", test_accuracy)

    # Confusion Matrix and Classification Report for Test Data
    conf_matrix = confusion_matrix(y_test, test_predictions, labels=class_names)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(classification_report(y_test, test_predictions, target_names=class_names))

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def save_model(svm_classifier, pca):
    joblib.dump(svm_classifier, 'svm_classifier.joblib')
    joblib.dump(pca, 'pca_transformer.joblib')

def load_model():
    svm_classifier = joblib.load('svm_classifier.joblib')
    pca_transformer = joblib.load('pca_transformer.joblib')
    return svm_classifier, pca_transformer

if __name__ == "__main__":
    train_model()

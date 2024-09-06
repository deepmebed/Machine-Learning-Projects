import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QFileDialog, QProgressBar
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from PIL import Image

# Load the trained models and label encoder
svm_classifier = joblib.load('svm_model.pkl')
random_forest_classifier = joblib.load('random_forest_model_1.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to extract HOG features from an image
def extract_hog_features(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    hog_features, hog_image = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return hog_features

# Function for predicting new images
def predict_image(image, model):
    image = image.resize((128, 128))
    features = extract_hog_features(image)
    features = features.reshape(1, -1)
    
    prediction = model.predict(features)
    label = label_encoder.inverse_transform(prediction)[0]
    return label

class IUBGateGuard(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('IUB GateGuard: Vehicle Access System')
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        main_layout = QVBoxLayout()

        # Title
        title = QLabel("IUB GateGuard: Vehicle Access System")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Authors
        authors_layout = QHBoxLayout()
        author1_label = QLabel("Yasir Hussain")
        author1_image = QLabel()
        author1_image.setPixmap(QPixmap("Yasir.png"))
        author2_label = QLabel("Syed Qasim Raza Fatmi")
        author2_image = QLabel()
        author2_image.setPixmap(QPixmap("Qasim.jpeg"))
        authors_layout.addWidget(author1_image)
        authors_layout.addWidget(author1_label)
        authors_layout.addWidget(author2_image)
        authors_layout.addWidget(author2_label)
        main_layout.addLayout(authors_layout)

        # Model selection
        model_label = QLabel("Select a model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["SVM", "Random Forest"])
        main_layout.addWidget(model_label)
        main_layout.addWidget(self.model_combo)

        # Image upload
        self.image_label = QLabel("Upload an image of the bike:")
        self.image_button = QPushButton("Select Image")
        self.image_button.clicked.connect(self.select_image)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.image_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.setLayout(main_layout)

    def select_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image files (*.jpg *.jpeg *.png)")
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                image_path = selected_files[0]
                image = Image.open(image_path)
                self.display_image(image)
                model = svm_classifier if self.model_combo.currentText() == "SVM" else random_forest_classifier
                label = predict_image(image, model)
                if label == 'bikes_with_stickers':
                    self.progress_bar.setValue(100)
                    self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
                    self.progress_bar.setFormat("✅ Vehicle permitted for campus entry: Sticker Affixed.")
                elif label == 'bikes_without_stickers':
                    self.progress_bar.setValue(100)
                    self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
                    self.progress_bar.setFormat("🚫 Unauthorized Vehicle Entry: Sticker Required for Campus Access.")
                else:
                    self.progress_bar.setValue(0)
                    self.progress_bar.setFormat("")

    def display_image(self, image):
        image = image.resize((400, 400))
        image_format = QImage.Format_RGB888 if image.mode == 'RGB' else QImage.Format_RGBA8888
        qt_image = QImage(image.tobytes(), image.width, image.height, image_format)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    iub_gate_guard = IUBGateGuard()
    iub_gate_guard.show()
    sys.exit(app.exec_())

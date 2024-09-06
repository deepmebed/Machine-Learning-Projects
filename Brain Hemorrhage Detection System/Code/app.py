
import numpy as np
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set a random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



# Method to convert image to vector by resizing
def image_to_vector(image, size):
    return cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC).flatten()

# Main method to extract features
def extract_features(images,  size=(32, 32)):
    return np.array([image_to_vector(image, size) for image in images])
  
       
# Shuffle and split data into train & test sets
def splitTestTrain(X, Y):
    trainSize = int(0.8 * X.shape[0])
    Y = np.reshape(Y, (Y.shape[0], 1))
    indexes = np.arange(X.shape[0])
    indexes = np.reshape(indexes, (X.shape[0], 1))
    data = np.concatenate((X, Y, indexes), axis=1)
    np.random.seed(RANDOM_SEED)  
    np.random.shuffle(data)
    trainX = data[:trainSize, :-2]
    trainY = data[:trainSize, -2]
    testX = data[trainSize:, :-2]
    testY = data[trainSize:, -2]
    imagesTest = data[trainSize:, -1]
    return trainX, trainY, testX, testY, imagesTest

# Main experiment
if __name__ == '__main__':
    # Load images and labels
    pathX = "../Dataset/mri/*.png"
    pathY = '../Dataset/labels.csv'
    files = sorted(glob.glob(pathX))
    labels_df = pd.read_csv(pathY)
    labels = np.array(labels_df[' hemorrhage'].tolist())

    # Define a consistent size for all images
    target_size = (256, 256)  # Adjust as needed

    # Load and resize images
    images = []
    for path in files:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, target_size)
            images.append(img_resized)
    images = np.array(images)
    print(f'Number of images loaded: {len(images)}')

    # Choose a method to extract features
    
    X = extract_features(images, size=target_size)

    # Split data into train & test sets, including shuffle of the data
    trainX, trainY, testX, testY, testIm = splitTestTrain(X, labels)

    # Print the number of training and testing images
    print(f'Number of training images: {trainX.shape[0]}')
    print(f'Number of testing images: {testX.shape[0]}')

    # Train the SVM model
    print('Training SVM model...')
    model = SVC(kernel='linear', random_state=RANDOM_SEED)  # You can change the kernel type and other hyperparameters as needed
    model.fit(trainX, trainY.ravel())

    # Predict and evaluate the model
    predictions = model.predict(testX)
    accuracy = accuracy_score(testY, predictions)
    train_predict = model.predict(trainX)

    train_accuracy = accuracy_score(trainY, train_predict )

    print('SVM accuracy: {:.2f}%'.format(accuracy * 100))
    # print('SVM train accuracy: {:.2f}%'.format(train_accuracy * 100))


    # Count how many test images have hemorrhage and how many do not
    hemorrhage_count = np.sum(predictions == 1)
    no_hemorrhage_count = np.sum(predictions == 0)

    print(f'Number of testing images with hemorrhage: {hemorrhage_count}')
    print(f'Number of testing images without hemorrhage: {no_hemorrhage_count}')



joblib.dump(model, 'trained_model.joblib')

# Draw the images and labels
def draw(images, labels):
    n = len(images)
    cols = 5
    rows = n // cols + int(n % cols > 0)
    plt.figure(figsize=(15, rows * 3))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))
        plt.title("Label: {}".format("Hemorrhage" if labels[i] == 1 else "No Hemorrhage"))
        plt.axis('off')
    plt.tight_layout()
    plt.show()


    # Draw the test images and their predicted labels (Optional)
    # draw(images[testIm.astype(int)], predictions)
 



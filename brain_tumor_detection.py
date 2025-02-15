import os
import cv2
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Define dataset path
DATA_DIR = "Dataset/brain_tumor_dataset"
CATEGORIES = ["no", "yes"]  # Match actual folder names

IMG_SIZE = 128  # Resize images to 128x128

# Load and preprocess dataset
def load_data():
    X, y = [], []
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        label = CATEGORIES.index(category)  # Convert category to numerical label (0 or 1)

        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize to (128,128)
                X.append(image.flatten())  # Flatten the image into a 1D array
                y.append(label)
            except Exception as e:
                print(f"Error loading image: {img_name}, Error: {e}")

    return np.array(X), np.array(y)

# Train and evaluate model
def train_model(X_train, y_train, X_test, y_test):
    model = SVC(kernel='linear')  # Support Vector Machine (SVM) with linear kernel
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Model Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=CATEGORIES))

    return model

# Predict a new image
def predict_brain_tumor(image_path, model):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.flatten().reshape(1, -1)  # Flatten and reshape for model input

    prediction = model.predict(image)
    result = "Tumor Detected" if prediction[0] == 1 else "No Tumor"
    
    # Display image with prediction
    plt.imshow(cv2.imread(image_path), cmap="gray")
    plt.title(result)
    plt.axis("off")
    plt.show()

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor Detection using SVM (Scikit-Learn)")
    parser.add_argument("--test_image", type=str, help="Path to a single MRI image for prediction")
    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset Loaded! Training: {len(X_train)}, Testing: {len(X_test)}\n")

    # Train model
    print("Training model...")
    trained_model = train_model(X_train, y_train, X_test, y_test)
    
    # Test with a single image if provided
    if args.test_image:
        print(f"\nPredicting on image: {args.test_image}")
        result = predict_brain_tumor(args.test_image, trained_model)
        print(f"Prediction Result: {result}")

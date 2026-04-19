import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load the Iris dataset once at the top of the file.
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

print("Iris dataset loaded successfully.")
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# --- Preprocessing ---
# Q1: Train/Test Split
# Split the dataset into training and testing sets.
# X = features (input data), y = target labels.
# test_size=0.2 means 20% of the data will be used for testing.
# stratify=y ensures that class distribution is preserved in both train and test sets.
# random_state=42 makes the split reproducible (same result every time you run the code).

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

# Print shapes to verify the split
# X_train and X_test contain feature data
# y_train and y_test contain corresponding labels

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q2: Feature Scaling

# Create a StandardScaler object.
# StandardScaler transforms each feature so it has mean ~0 and standard deviation ~1.
scaler = StandardScaler()

# Fit the scaler only on the training data.
# This prevents data leakage because the test set must remain unseen during training.
scaler.fit(X_train)
# The scaler is fit only on X_train to avoid leaking information from the test set into the training process.

# Use the fitted scaler to transform both training and test features.
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the mean of each column in the scaled training data.
# These values should be very close to 0 after standardization.
print("Column means in X_train_scaled:")
print(X_train_scaled.mean(axis=0))

# Why do we fit scaler only on training data? 
# To avoid data leakage. If we fit on the full dataset, the model indirectly learns information from the test set, 
# which leads to overly optimistic performance.

# --- KNN ---
# Q1: Train KNN on unscaled data

# Create a KNN classifier with k=5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model on the unscaled training data
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
print("KNN (unscaled data) accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Q2: Train KNN on scaled data

# Create a new KNN model
knn_scaled = KNeighborsClassifier(n_neighbors=5)

# Fit on scaled training data
knn_scaled.fit(X_train_scaled, y_train)

# Predict on scaled test data
y_pred_scaled = knn_scaled.predict(X_test_scaled)

# Evaluate accuracy
print("\nKNN (scaled data) accuracy:", accuracy_score(y_test, y_pred_scaled))

# In this case, scaling does not significantly change performance because the Iris dataset
# is already well-behaved and features are on similar scales. However, in general, KNN
# benefits from scaling because it relies on distance calculations.
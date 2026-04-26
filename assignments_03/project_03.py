import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

print("Project 03 setup completed successfully.")

# --- Task 1: Load and Explore ---

base_dir = os.path.dirname(__file__)

data_path = os.path.join(
    base_dir,
    "..",
    "resources",
    "spambase",
    "spambase.data"
)

df = pd.read_csv(data_path, header=None)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)
print(df.head())

# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nClass distribution:")
print(y.value_counts())

print("\nClass proportions:")
print(y.value_counts(normalize=True))

# The dataset is moderately imbalanced.
# Non-spam emails make up about 60%, while spam emails are about 40%.
# This means accuracy alone may not be a reliable metric.

# --- Task 2: Train/Test Split and Baseline ---

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain/Test shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Baseline model: predict all zeros (non-spam)
y_pred_baseline = np.zeros_like(y_test)

# Evaluate baseline
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print("\nBaseline Accuracy:", baseline_accuracy)

print("\nBaseline Classification Report:")
print(classification_report(y_test, y_pred_baseline))

# The baseline model achieves around 58% accuracy by predicting all emails as non-spam. 
# However, it completely fails to detect spam, which makes it ineffective. 
# This demonstrates that accuracy alone is not a reliable metric for imbalanced classification problems.

# --- Task 3: KNN Classification ---

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # Using K=5 as a common default choice
knn.fit(X_train, y_train)

# Predict
y_pred_knn = knn.predict(X_test)

# Evaluate
print("\nKNN Accuracy:", accuracy_score(y_test, y_pred_knn))

print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

# KNN significantly improves performance compared to the baseline.
# It is able to detect spam emails with a recall of around 71%.
# However, some spam emails are still missed, indicating room for improvement.

# --- Task 4: KNN with Scaling ---

# Scaling is important for KNN because it relies on distance calculations.
# Features with larger ranges can dominate the distance if not scaled.

scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN again
knn_scaled = KNeighborsClassifier(n_neighbors=5)  # Using the same K value for consistency
knn_scaled.fit(X_train_scaled, y_train)

# Predict
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

# Evaluate
print("\nKNN (Scaled) Accuracy:", accuracy_score(y_test, y_pred_knn_scaled))

print("\nKNN (Scaled) Classification Report:")
print(classification_report(y_test, y_pred_knn_scaled))

# Applying feature scaling significantly improved KNN performance.
# The model now detects spam emails more effectively (recall ~84%)
# while maintaining high precision (~90%).
# This confirms that scaling is important for distance-based models like KNN.
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
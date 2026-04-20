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
from sklearn.multiclass import OneVsRestClassifier

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
# is already well-behaved and features are on similar scales. After scaling, distances slightly change, 
# which can lead to minor performance drops. 
# However, in general, KNN benefits from scaling because it relies on distance calculations.

# Q3: Cross-validation for KNN (unscaled data)

# Create KNN model
knn_cv = KNeighborsClassifier(n_neighbors=5)

# Perform 5-fold cross-validation on training data
cv_scores = cross_val_score(knn_cv, X_train, y_train, cv=5)

# Print each fold score
print("\nCross-validation scores:", cv_scores)

# Print mean and standard deviation
print("Mean CV score:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())

# Cross-validation is more reliable than a single train/test split
# because it evaluates the model on multiple subsets of the data,
# reducing the impact of randomness in how the data is split.

# Q4: Hyperparameter tuning for KNN (choose best k)

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

print("\nK values and corresponding mean CV scores:")

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    
    scores = cross_val_score(knn_model, X_train, y_train, cv=5)
    
    print(f"k = {k}, Mean CV Score = {scores.mean():.4f}")

# We select the k value that gives the highest mean cross-validation score,
# as it is expected to generalize best to unseen data. I had two k values (5 and 7) with the same mean CV score, 
# so I chose k=5 because it is simpler and less likely to overfit compared to k=7 (k=5 achieves the highest mean cross-validation score.
# Although k=7 has the same score, we prefer a smaller k to capture more local patterns in the data.)

# --- Classifier Evaluation ---
# Q1: Confusion Matrix

# Create confusion matrix using predictions from unscaled KNN
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()

# Save the plot
plt.savefig("outputs/knn_confusion_matrix.png")

plt.show()

# The confusion matrix shows how often the model correctly or incorrectly classifies each class.
# In this case, there are likely very few or no misclassifications due to high accuracy.
# The confusion matrix shows perfect classification with no misclassifications.
# All predictions fall on the diagonal, indicating that the model correctly classifies all samples.
# This suggests that the dataset is well-separated and easy for KNN to model. 

# --- Decision Trees ---
# Q1: Train Decision Tree

# Create a Decision Tree classifier with limited depth
dt = DecisionTreeClassifier(max_depth=3, random_state=42)

# Fit the model on unscaled data (scaling not required for trees)
dt.fit(X_train, y_train)

# Predict on test data
y_pred_dt = dt.predict(X_test)

# Evaluate performance
print("\nDecision Tree accuracy:", accuracy_score(y_test, y_pred_dt))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# The Decision Tree achieves slightly lower accuracy compared to KNN.
# This is likely because the dataset is small and well-separated, where KNN performs very well.
# However, Decision Trees provide interpretable rules, which can be useful in real-world scenarios.

# Since Decision Trees do not rely on distance calculations, scaling the data does not affect their performance.

# --- Logistic Regression ---
# Q1: Regularization effect

# Test how different C values affect coefficient magnitude.
# Smaller C means stronger regularization.
# Larger C means weaker regularization.
C_values = [0.01, 1.0, 100]

for C in C_values:
    # Wrap LogisticRegression in OneVsRestClassifier because
    # the liblinear solver in this environment does not directly
    # support multiclass classification for the Iris dataset.
    model = OneVsRestClassifier(
        LogisticRegression(C=C, max_iter=1000, solver='liblinear')
    )

    # Fit the model on scaled training data
    model.fit(X_train_scaled, y_train)

    # Each binary classifier has its own coefficient array.
    # Sum the absolute values across all class-specific estimators.
    coef_sum = sum(np.abs(estimator.coef_).sum() for estimator in model.estimators_)

    print(f"C = {C}, Total Coefficient Magnitude = {coef_sum:.4f}")

# As C increases, the total coefficient magnitude usually increases.
# This shows that weaker regularization allows the model to use larger weights.
# Smaller C applies stronger regularization and keeps coefficients smaller.

# --- PCA ---
# Load the handwritten digits dataset for PCA experiments.
digits = load_digits()
X_digits = digits.data      # Shape: (1797, 64), each image flattened into 64 features
y_digits = digits.target    # Digit labels: 0 through 9
images = digits.images      # Same images in 8x8 matrix form for plotting

# Q1: Print dataset shapes
print("\nX_digits shape:", X_digits.shape)
print("images shape:", images.shape)

# Create one example image for each digit class (0-9)
fig, axes = plt.subplots(1, 10, figsize=(15, 3))

for digit in range(10):
    # Find the first index where the target label matches the digit
    first_index = np.where(y_digits == digit)[0][0]

    # Display the corresponding 8x8 image
    axes[digit].imshow(images[first_index], cmap="gray_r")
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
plt.show()

# Q2: PCA 2D projection

# Create PCA object (no limit on number of components)
pca = PCA()

# Fit PCA on the digits data
pca.fit(X_digits)

# Transform the data into principal component space
scores = pca.transform(X_digits)

# Plot first two principal components
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    scores[:, 0], 
    scores[:, 1], 
    c=y_digits, 
    cmap="tab10", 
    s=10
)

plt.colorbar(scatter, label="Digit")
plt.title("PCA 2D Projection of Digits Dataset")

plt.savefig("outputs/pca_2d_projection.png")
plt.show()

# In the PCA projection, samples with the same digit label often form clusters,
# indicating that PCA captures meaningful structure in the data.
# Samples with the same digit label tend to cluster together,
# although there is some overlap between classes.
# This indicates that PCA captures meaningful structure,
# but 2 components are not enough for perfect separation.

# Q3: Explained variance

# Compute cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative variance
plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")

plt.grid()

plt.savefig("outputs/pca_variance_explained.png")
plt.show()

# The cumulative explained variance shows how much information is retained
# as we increase the number of principal components.
# We typically choose the number of components that explain around 80–90% of the variance.

# Approximately 12–13 components are needed to explain around 80% of the variance.
# This shows that we can significantly reduce dimensionality from 64 to about 12
# while retaining most of the information.

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

# Q4: PCA Reconstruction

n_values = [2, 5, 15, 40]

fig, axes = plt.subplots(len(n_values) + 1, 5, figsize=(10, 8))

# Original images (first row)
for i in range(5):
    axes[0, i].imshow(images[i], cmap="gray_r")
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")

# Reconstructed images
for row_idx, n in enumerate(n_values):
    for col_idx in range(5):
        reconstructed = reconstruct_digit(col_idx, scores, pca, n)
        axes[row_idx + 1, col_idx].imshow(reconstructed, cmap="gray_r")
        axes[row_idx + 1, col_idx].set_title(f"n={n}")
        axes[row_idx + 1, col_idx].axis("off")

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.show()

# As the number of components increases, the reconstructed images become clearer.
# Around 15 components, digits become recognizable, which aligns with the explained variance curve.
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

print("Project 03 setup completed successfully.")

# --- Task 1: Load and Explore ---

os.makedirs("outputs", exist_ok=True)

base_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(
    base_dir,
    "..",
    "resources",
    "spambase",
    "spambase.data"
)

# Column names from spambase.names (UCI ML Repository).
# 48 word frequency features, 6 character frequency features,
# 3 capital run length features, and 1 target label.
column_names = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
    "char_freq_$", "char_freq_#",
    "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total", "spam_label"
]

df = pd.read_csv(data_path, header=None, names=column_names)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)
print(df.head())

X = df.drop(columns=["spam_label"])
y = df["spam_label"]

print("\nClass distribution:")
print(y.value_counts())

print("\nClass proportions:")
print(y.value_counts(normalize=True))

# The dataset is moderately imbalanced: ~60% ham, ~40% spam.
# A naive classifier that always predicts "ham" achieves ~60% accuracy.
# This means raw accuracy can be misleading — we should also look at
# precision, recall, and F1-score to understand model quality.

# --- Baseline Model ---

# Baseline: always predict non-spam (the majority class).
# This is the floor any useful model must beat.
y_pred_baseline = np.zeros_like(y)

print("\nBaseline Accuracy on Full Dataset:", accuracy_score(y, y_pred_baseline))

print("\nBaseline Classification Report on Full Dataset:")
print(classification_report(y, y_pred_baseline, zero_division=0))

# The baseline achieves 60% accuracy but completely fails on spam (class 1).
# Any meaningful classifier must substantially outperform this.

# --- Boxplots: Task 1 ---

# We plot three features that are expected to differ clearly between spam and ham.
# - word_freq_free: spam often contains words like "free"
# - char_freq_!: spam tends to use exclamation marks heavily
# - capital_run_length_total: spam often has long bursts of capital letters

features_to_plot = ["word_freq_free", "char_freq_!", "capital_run_length_total"]

for feature in features_to_plot:
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column=feature, by="spam_label", ax=ax)
    ax.set_title(f"Distribution of '{feature}' by Class")
    ax.set_xlabel("Class (0 = Ham, 1 = Spam)")
    ax.set_ylabel(feature)
    plt.suptitle("")  # Remove the default "Boxplot grouped by" title
    plt.tight_layout()
    safe_name = feature.replace("!", "excl").replace("$", "dollar")
    plt.savefig(f"outputs/boxplot_{safe_name}.png")
    plt.close()
    print(f"Saved boxplot for '{feature}' to outputs/")

# What the boxplots reveal:
# - word_freq_free: spam emails contain this word far more often; ham emails rarely have it.
# - char_freq_!: spam uses exclamation marks much more frequently.
# - capital_run_length_total: spam tends to have dramatically longer capital runs.
# All three features show strong class separation, confirming they are useful predictors.

# Feature scale observation:
# Many emails have a value of zero for most word-frequency features —
# most emails simply do not contain words like "free" at all.
# This heavy skew toward zero is common in text-derived features and creates
# a sparse distribution that can affect distance-based models like KNN.
# The numeric scale also varies wildly: word frequencies are tiny fractions (0–1),
# while capital_run_length_total can reach into the thousands.
# This scale mismatch matters for KNN and logistic regression, which use distances
# or coefficient magnitudes. Decision trees and random forests are unaffected
# because they split on feature thresholds, not distances.

# --- Task 2: Prepare Your Data ---

# Split dataset into training and testing sets.
# random_state=42 ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain/Test shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Scale features for distance-based and coefficient-based models.
# The scaler is fit only on training data to prevent data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA must be applied after scaling.
# Without scaling, features with large raw values (like capital_run_length_total)
# dominate the principal components, making PCA unreliable.
# We fit PCA on training data only — same reason as the scaler.
pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance)
plt.axhline(y=0.90, color="r", linestyle="--", label="90% threshold")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/pca_cumulative_explained_variance.png")
plt.close()

# Find the first number of components that explains at least 90% of variance.
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print("\nNumber of PCA components for 90% variance:", n_components_90)

# Transform scaled train and test data into PCA space (first n components).
X_train_pca = pca.transform(X_train_scaled)[:, :n_components_90]
X_test_pca = pca.transform(X_test_scaled)[:, :n_components_90]

print("X_train_pca shape:", X_train_pca.shape)
print("X_test_pca shape:", X_test_pca.shape)

# --- Task 3: Classifier Comparison ---

# 1. KNN on unscaled data
# Without scaling, capital_run_length_total (values up to thousands)
# dominates the Euclidean distance, drowning out word frequency features.
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_knn_unscaled = knn_unscaled.predict(X_test)

print("\nKNN (Unscaled) Accuracy:", accuracy_score(y_test, y_pred_knn_unscaled))
print("\nKNN (Unscaled) Classification Report:")
print(classification_report(y_test, y_pred_knn_unscaled))

# 2. KNN on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

print("\nKNN (Scaled) Accuracy:", accuracy_score(y_test, y_pred_knn_scaled))
print("\nKNN (Scaled) Classification Report:")
print(classification_report(y_test, y_pred_knn_scaled))

# 3. KNN on PCA-reduced data
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_knn_pca = knn_pca.predict(X_test_pca)

print("\nKNN (PCA) Accuracy:", accuracy_score(y_test, y_pred_knn_pca))
print("\nKNN (PCA) Classification Report:")
print(classification_report(y_test, y_pred_knn_pca))

# Scaling had a large positive effect on KNN (from ~79% to ~89%).
# PCA gave a small additional improvement, suggesting the 57 features
# contain moderate redundancy but are mostly informative.

# Hyperparameter tuning: K values on scaled data
k_values = [1, 3, 5, 7, 9, 11, 15]
print("\nKNN k-value comparison (5-fold CV on scaled training data):")
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=5)
    print(f"  k = {k:2d}, Mean CV Score = {scores.mean():.4f}")

# 4. Decision Tree: depth comparison
# Trees split on feature thresholds, so they do not need scaling.
depth_values = [3, 5, 10, None]
print("\nDecision Tree depth comparison:")
for depth in depth_values:
    tree_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_model.fit(X_train, y_train)
    train_acc = tree_model.score(X_train, y_train)
    test_acc = tree_model.score(X_test, y_test)
    print(
        f"  max_depth = {str(depth):4s}, "
        f"Train Accuracy = {train_acc:.4f}, "
        f"Test Accuracy = {test_acc:.4f}"
    )

# As depth increases, training accuracy rises toward 1.0 while test accuracy
# peaks around depth=10 and then drops slightly at None (unlimited).
# This is a classic overfitting pattern: unlimited depth memorizes training
# examples instead of learning general patterns.
# Chosen depth: 10 — it achieves the best test accuracy and a reasonable
# train/test gap, indicating the model generalizes well without memorizing.

best_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
best_tree.fit(X_train, y_train)
y_pred_tree = best_tree.predict(X_test)

print("\nDecision Tree (depth=10) Accuracy:", accuracy_score(y_test, y_pred_tree))
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))

# 5. Random Forest
# Random Forest trains many trees on random subsets of data and features,
# then averages their predictions. This reduces overfitting vs. a single tree
# and typically produces more stable, accurate results.
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# 6. Feature importances: Decision Tree vs Random Forest
tree_importances = pd.Series(best_tree.feature_importances_, index=X.columns)
rf_importances = pd.Series(rf.feature_importances_, index=X.columns)

print("\nTop 10 Decision Tree Feature Importances:")
print(tree_importances.sort_values(ascending=False).head(10))

print("\nTop 10 Random Forest Feature Importances:")
print(rf_importances.sort_values(ascending=False).head(10))

# Both models agree that char_freq_$, char_freq_!, word_freq_remove,
# and capital run length features are the strongest predictors.
# This matches intuition: spam uses dollar signs, exclamation marks,
# the word "remove" (unsubscribe prompts), and long capital letter runs.
# The Decision Tree concentrates importance in fewer features (one feature
# can dominate early splits), while the Random Forest distributes it more
# evenly — a sign of a more robust, less brittle representation.

top_rf_importances = rf_importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
top_rf_importances.sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.close()

# 7. Logistic Regression on scaled data
log_reg_scaled = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
log_reg_scaled.fit(X_train_scaled, y_train)
y_pred_log_scaled = log_reg_scaled.predict(X_test_scaled)

print("\nLogistic Regression (Scaled) Accuracy:", accuracy_score(y_test, y_pred_log_scaled))
print("\nLogistic Regression (Scaled) Classification Report:")
print(classification_report(y_test, y_pred_log_scaled))

# 8. Logistic Regression on PCA-reduced data
log_reg_pca = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
log_reg_pca.fit(X_train_pca, y_train)
y_pred_log_pca = log_reg_pca.predict(X_test_pca)

print("\nLogistic Regression (PCA) Accuracy:", accuracy_score(y_test, y_pred_log_pca))
print("\nLogistic Regression (PCA) Classification Report:")
print(classification_report(y_test, y_pred_log_pca))

# Logistic Regression performed slightly worse with PCA than with full scaled data.
# This suggests that PCA removed some information useful for the linear decision
# boundary — the ~10% of variance dropped by PCA was not purely noise.

# --- Best Model: Confusion Matrix ---

# Random Forest is the best-performing model based on accuracy and F1-score.
best_model_name = "Random Forest"
best_model_predictions = y_pred_rf

best_cm = confusion_matrix(y_test, best_model_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=best_cm,
    display_labels=["Ham", "Spam"]
)
disp.plot()
plt.title(f"{best_model_name} Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.close()

print(f"\nBest model selected: {best_model_name}")
print("Confusion matrix (rows = actual, cols = predicted):")
print(best_cm)

# The confusion matrix shows that false negatives (spam classified as ham)
# are more frequent than false positives (ham classified as spam).
# The model is conservative: it rarely flags a legitimate email as spam.

# --- Task 3 Summary ---

# Model performance ranking (approx. test accuracy):
#   Random Forest              ~95.5%   <-- best
#   Decision Tree (depth=10)   ~92.5%
#   Logistic Regression        ~91.9%
#   KNN (PCA)                  ~89.7%
#   KNN (Scaled)               ~89.4%
#   KNN (Unscaled)             ~79.0%   <-- worst

# Random Forest wins because it combines 100 trees trained on random subsets,
# reducing both overfitting and variance compared to a single tree.

# KNN improved dramatically with scaling (79% → 89%), confirming that
# unscaled distance calculations are dominated by large-range features.
# PCA gave a small further improvement for KNN, but slightly hurt Logistic
# Regression, where it discarded informative variance that the model could
# otherwise use. This matched the hypothesis from Task 2: KNN can benefit
# from dimensionality reduction to avoid the curse of dimensionality, while
# logistic regression handles correlated features well via regularization.

# For a spam filter, accuracy alone is not enough.
# False positives (legitimate email marked as spam) are more disruptive to
# users than false negatives (spam that slips through). Receiving an extra
# spam message is a minor annoyance; losing an important email to the spam
# folder is a real cost. Therefore we want high precision for the spam class —
# when the model predicts spam, it should almost always be right — even if
# that means letting some spam through. The Random Forest achieves this well:
# ~0.98 precision for spam with a low false-positive rate of only 9 emails.
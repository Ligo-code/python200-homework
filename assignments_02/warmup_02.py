import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# OUTPUT_DIR is always relative to this script's location,
# regardless of where the script is run from.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- scikit-learn API ---

# Q1
# Core pattern: create -> fit -> predict
# We have years of experience and salaries for 6 employees.
# Goal: train a model, then predict salaries for 4 and 8 years.

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model_q1 = LinearRegression()
model_q1.fit(years, salary)

pred_4yr = model_q1.predict([[4]])[0]
pred_8yr = model_q1.predict([[8]])[0]

print("=== Q1: Linear Regression (years vs salary) ===")
print(f"Slope (coef):      {model_q1.coef_[0]:.2f}")
print(f"Intercept:         {model_q1.intercept_:.2f}")
print(f"Predicted salary at 4 years: ${pred_4yr:,.0f}")
print(f"Predicted salary at 8 years: ${pred_8yr:,.0f}")

# Q2
# scikit-learn requires X to be 2D because its API is designed to handle
# multiple features at once. Even with one feature, it expects each row
# to be a sample and each column to be a feature. A 1D array is ambiguous:
# it could be a single sample with many features, or many samples with one
# feature. reshape(-1, 1) makes it unambiguously "many samples, 1 feature".

x = np.array([10, 20, 30, 40, 50])
print("\n=== Q2: Reshaping X to 2D ===")
print(f"Original shape: {x.shape}")
x_2d = x.reshape(-1, 1)
print(f"Reshaped shape: {x_2d.shape}")

# Q3
# K-Means is unsupervised - it finds structure without labels.
# Same API: create -> fit -> predict.

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print("\n=== Q3: K-Means Clustering ===")
print("Cluster centers:")
print(kmeans.cluster_centers_)
print("Points per cluster:", np.bincount(labels))

plt.figure()
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis", s=40)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="black", marker="X", s=150, label="Centers"
)
plt.title("K-Means Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_clusters.png"))
plt.close()
print(f"Plot saved: {os.path.join(OUTPUT_DIR, 'kmeans_clusters.png')}")

# --- Linear Regression ---

# Synthetic medical dataset: 100 patients, age + smoker flag -> annual cost.
# Generated once and reused across all Linear Regression questions.

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# LR Q1
# Scatter plot: age vs cost, colored by smoker status.
# c=smoker maps 0 (non-smoker) to cool blue and 1 (smoker) to warm red.

print("\n=== LR Q1: Cost vs Age scatter plot ===")

plt.figure()
plt.scatter(age, cost, c=smoker, cmap="coolwarm", s=40)
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Annual Medical Cost ($)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cost_vs_age.png"))
plt.close()
print(f"Plot saved: {os.path.join(OUTPUT_DIR, 'cost_vs_age.png')}")
# Two distinct horizontal bands are visible: the upper band (red/warm) is
# smokers, the lower band (blue/cool) is non-smokers. Both bands trend
# upward with age. This tells us smoker status is a strong predictor --
# it shifts costs up by a roughly constant amount regardless of age.

# LR Q2
# Train/test split: 80% train, 20% test.
# We use only 'age' as the feature here (single-feature model).

X_age = age.reshape(-1, 1)
y_cost = cost

X_train, X_test, y_train, y_test = train_test_split(
    X_age, y_cost, test_size=0.2, random_state=42
)

print("\n=== LR Q2: Train/Test Split shapes ===")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# LR Q3
# Fit a single-feature model (age only) and evaluate on the test set.

model_age = LinearRegression()
model_age.fit(X_train, y_train)

y_pred_age = model_age.predict(X_test)
rmse_age   = np.sqrt(np.mean((y_pred_age - y_test) ** 2))
r2_age     = model_age.score(X_test, y_test)

print("\n=== LR Q3: Single-feature model (age only) ===")
print(f"Slope (age coef):  {model_age.coef_[0]:.2f}")
print(f"Intercept:         {model_age.intercept_:.2f}")
print(f"RMSE:              {rmse_age:,.2f}")
print(f"R2 (test):         {r2_age:.4f}")
# The slope ~200 means each additional year of age adds roughly $200 to
# annual medical costs -- matching the true data-generating relationship.
# However, because we ignored the smoker flag, predictions for smokers
# and non-smokers are blended together, which inflates RMSE and lowers R2.

# LR Q4
# Add smoker as a second feature. The model now has two coefficients.

X_full = np.column_stack([age, smoker])
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y_cost, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

y_pred_full = model_full.predict(X_test_f)
r2_full   = model_full.score(X_test_f, y_test_f)
rmse_full = np.sqrt(np.mean((y_pred_full - y_test_f) ** 2))

print("\n=== LR Q4: Two-feature model (age + smoker) ===")
print(f"age coefficient:    {model_full.coef_[0]:.2f}")
print(f"smoker coefficient: {model_full.coef_[1]:.2f}")
print(f"R2 (test, age only):   {r2_age:.4f}")
print(f"R2 (test, age+smoker): {r2_full:.4f}")
print(f"RMSE (age+smoker):     {rmse_full:,.2f}")
# The smoker coefficient (~15000) represents the average extra annual cost
# for a smoker versus a non-smoker of the same age. Adding the smoker flag
# dramatically improves R2 because it captures the two-band structure
# visible in the scatter plot.

# LR Q5
# Predicted vs Actual plot for the two-feature model.
# Perfect predictions would lie on the diagonal (y = x).
# y_pred_full was already computed in Q4.

plt.figure()
plt.scatter(y_pred_full, y_test_f, alpha=0.7, s=40)
diag_min = min(y_pred_full.min(), y_test_f.min())
diag_max = max(y_pred_full.max(), y_test_f.max())
plt.plot([diag_min, diag_max], [diag_min, diag_max], "r--", label="Perfect fit")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost ($)")
plt.ylabel("Actual Cost ($)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "predicted_vs_actual.png"))
plt.close()

print("\n=== LR Q5: Predicted vs Actual plot ===")
print(f"Plot saved: {os.path.join(OUTPUT_DIR, 'predicted_vs_actual.png')}")
# A point ABOVE the diagonal means the actual cost was higher than predicted
# -- the model underestimated (e.g. a heavy smoker with extra complications).
# A point BELOW the diagonal means the model overestimated the cost.
# Random scatter around the line is normal noise; systematic curves or
# clusters indicate the model is missing an important pattern.

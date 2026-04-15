import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The file uses semicolons as separators, not commas.
# pd.read_csv("file.csv") without sep=";" would read the entire row
# as a single column -- every value would be wrong.
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "resources", "student_performance_math.csv")

# =============================================================================
# Task 1: Load and Explore
# =============================================================================

df = pd.read_csv(DATA_PATH, sep=";")

print("=== Task 1: Load and Explore ===")
print(f"Shape: {df.shape}")
print()
print("First 5 rows:")
print(df.head())
print()
print("Data types:")
print(df.dtypes)

# Histogram of G3 with 21 bins (one per possible grade value 0-20).
# We expect to see a cluster of zeros sitting apart from the main distribution --
# those are students who did not take the final exam.
plt.figure()
plt.hist(df["G3"], bins=21, range=(-0.5, 20.5), edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("G3 (Final Grade)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "g3_distribution.png"))
plt.close()
print(f"\nPlot saved: {os.path.join(OUTPUT_DIR, 'g3_distribution.png')}")

# =============================================================================
# Task 2: Preprocess the Data
# =============================================================================

print("\n=== Task 2: Preprocess ===")

# --- Step 1: Filter out G3=0 rows ---
# G3=0 does not mean a student scored zero -- it means they were absent
# from the final exam. Keeping these rows would teach the model a false
# pattern: it would try to predict "exam absence" as if it were a grade,
# mixing two completely different phenomena into one target variable.
print(f"Shape before filtering G3=0: {df.shape}")
df_clean = df[df["G3"] > 0].copy()
print(f"Shape after  filtering G3=0: {df_clean.shape}")
removed     = len(df) - len(df_clean)
removed_pct = removed / len(df) * 100
print(f"Rows removed: {removed} ({removed_pct:.2f}%)")

# --- Step 2: Convert yes/no columns to 1/0 ---
yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

# Sanity check: if any unexpected value (e.g. "Yes", "NO") slipped through,
# map() would silently produce NaN instead of raising an error.
nan_counts = df_clean[yes_no_cols].isna().sum()
print(f"\nNaN counts after yes/no encoding:\n{nan_counts}")

# --- Step 3: Convert sex column F/M to 0/1 ---
# F=0, M=1. The dataset is from Portugal 2005; male students show a modest
# math advantage that reflects the social context, not an inherent difference.
df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

# --- Step 4: Compare absences correlation before and after filtering ---
# This is a striking result worth examining.
corr_before = df["absences"].corr(df["G3"])
corr_after  = df_clean["absences"].corr(df_clean["G3"])

print(f"\nPearson corr(absences, G3) BEFORE filtering: {corr_before:.4f}")
print(f"Pearson corr(absences, G3) AFTER  filtering: {corr_after:.4f}")
# Before filtering, students with G3=0 had high absences AND G3=0 (by
# definition -- they skipped the exam). This created an artificial strong
# negative correlation between absences and G3. After removing them, the
# true (much weaker) signal remains: absences alone is a poor predictor
# of how well a student who actually showed up will perform.

print("\nAll dtypes after preprocessing:")
print(df_clean.dtypes)

print("\nG3 stats after cleaning:")
print(df_clean["G3"].describe())

# =============================================================================
# Task 3: Exploratory Data Analysis
# =============================================================================

print("\n=== Task 3: EDA ===")

# Compute Pearson correlation between each numeric feature and G3.
# We exclude G1 and G2 -- they are near-perfect predictors and would
# make the analysis trivial (the task explicitly forbids using them).
feature_cols_eda = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "absences", "freetime", "goout", "Walc",
    "sex", "schoolsup", "internet", "higher", "activities"
]

correlations = (
    df_clean[feature_cols_eda + ["G3"]]
    .corr()["G3"]
    .drop("G3")
    .sort_values()
)

print("\nPearson correlation with G3 (sorted):")
print(correlations.round(4).to_string())

print("\nTop 3 positive correlations:")
print(correlations.sort_values(ascending=False).head(3).round(4))

print("\nTop 3 negative correlations:")
print(correlations.head(3).round(4))
# failures has the strongest negative correlation -- more past failures,
# lower final grade. Medu (mother's education) is the strongest positive
# signal: family educational background matters.
# Surprising: age is slightly negative -- older students in secondary
# school may have repeated years, meaning they are already struggling.
# schoolsup is negative due to selection bias: weak students are more
# likely to receive extra support, not because support hurts performance.

# --- Plot 1: failures vs G3 (box plot) ---
# Box plot is ideal for a discrete feature (0, 1, 2, 3) vs a continuous target.
# It shows median, spread, and outliers for each failure count group.
plt.figure()
failure_labels = sorted(df_clean["failures"].unique())
failure_groups = [df_clean[df_clean["failures"] == k]["G3"].values for k in failure_labels]
plt.boxplot(failure_groups, labels=[str(k) for k in failure_labels])
plt.title("G3 by Number of Past Failures")
plt.xlabel("Past Failures")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "g3_by_failures.png"))
plt.close()
print(f"\nPlot saved: {os.path.join(OUTPUT_DIR, 'g3_by_failures.png')}")
# Clear downward trend: students with 0 failures have the highest and most
# consistent grades. Each additional failure group has a lower median and
# more spread. Students with 3 failures have a very wide range -- some
# recover, others do not.

# --- Plot 2: higher (wants higher education) vs G3 (box plot) ---
# This binary feature has the strongest positive correlation with G3.
# A box plot by group reveals how large the gap actually is.
plt.figure()
groups_higher = [
    df_clean[df_clean["higher"] == 0]["G3"].values,
    df_clean[df_clean["higher"] == 1]["G3"].values,
]
plt.boxplot(groups_higher, labels=["No (0)", "Yes (1)"])
plt.title("G3 by Aspiration for Higher Education")
plt.xlabel("Wants Higher Education")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "g3_by_higher.png"))
plt.close()
print(f"Plot saved: {os.path.join(OUTPUT_DIR, 'g3_by_higher.png')}")
# Students who want higher education score notably higher on average.
# This likely reflects motivation and study habits rather than ability --
# the aspiration drives the behavior that produces the grade.

# --- Plot 3: G3 vs absences (scatter) ---
# Scatter shows the real shape of the relationship and reveals noise/outliers.
print("\nAbsences stats:")
print(df_clean["absences"].describe())
# There are extreme outliers (50+ absences). These students are rare but
# may pull the regression line. The relationship is noisy -- most students
# cluster near 0-10 absences with a wide range of grades.

plt.figure()
plt.scatter(df_clean["absences"], df_clean["G3"], alpha=0.5, s=30)
plt.title("G3 vs Absences")
plt.xlabel("Absences")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "g3_vs_absences.png"))
plt.close()
print(f"Plot saved: {os.path.join(OUTPUT_DIR, 'g3_vs_absences.png')}")
# The scatter confirms a weak negative trend with heavy noise.

# --- Plot 4: Correlation matrix heatmap ---
# Shows not just feature-vs-G3 but also feature-vs-feature relationships.
# For example, Medu and Fedu likely correlate with each other (parents
# tend to have similar education levels), which is worth knowing before
# interpreting individual coefficients.
plt.figure()
corr_matrix = df_clean[feature_cols_eda + ["G3"]].corr()
plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"))
plt.close()
print(f"Plot saved: {os.path.join(OUTPUT_DIR, 'correlation_matrix.png')}")
# Medu and Fedu show moderate positive correlation with each other --
# parents from similar educational backgrounds. goout and Walc also
# correlate (going out and drinking tend to go together).

# =============================================================================
# Task 4: Baseline Model
# =============================================================================

print("\n=== Task 4: Baseline Model (failures only) ===")

# The simplest possible model: one feature, no tuning.
# This gives us a floor to beat -- any more complex model should do better.
X_base = df_clean[["failures"]].values
y      = df_clean["G3"].values

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)

model_base = LinearRegression()
model_base.fit(X_train_b, y_train_b)

y_pred_b   = model_base.predict(X_test_b)
rmse_base  = np.sqrt(np.mean((y_pred_b - y_test_b) ** 2))
r2_base    = model_base.score(X_test_b, y_test_b)

print(f"Slope (failures coef): {model_base.coef_[0]:.4f}")
print(f"Intercept:             {model_base.intercept_:.4f}")
print(f"RMSE:                  {rmse_base:.4f}")
print(f"R2 (test):             {r2_base:.4f}")
# The slope is negative (~-1.5 to -2): each additional past failure is
# associated with roughly 1.5-2 points lower on the 0-20 scale.
# RMSE ~3+ means typical predictions are off by 3 grade points on a 0-20
# scale -- not great, but this is a single-feature model.
# R2 ~0.08 means failures alone explains only ~8% of the variance in G3.
# This is only slightly better than predicting the mean, so failures
# captures a real but limited signal and is far from sufficient on its own.
# we saw in EDA (-0.29), but far from enough for a useful model.
# The wide spread of predictions shows that many students with the same
# number of failures still achieve very different grades, indicating that
# other factors (study habits, motivation, support) play a major role.

# Compare against the dumbest possible baseline: always predict the mean.
# If our model can't beat this, it's useless.
y_mean     = np.mean(y_train_b)
rmse_mean  = np.sqrt(np.mean((y_mean - y_test_b) ** 2))
print(f"RMSE (predict mean): {rmse_mean:.4f}")
print(f"RMSE (failures):     {rmse_base:.4f}")
# Our model beats the mean-only baseline, but not by much --
# a sign that failures alone captures a real but weak signal.

plt.figure()
plt.scatter(X_test_b, y_test_b, alpha=0.7, label="Actual", s=40)
plt.scatter(X_test_b, y_pred_b, alpha=0.7, label="Predicted", marker="x", s=60)
plt.title("Baseline Model: Failures vs G3")
plt.xlabel("Failures")
plt.ylabel("G3")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "baseline_failures.png"))
plt.close()
print(f"Plot saved: {os.path.join(OUTPUT_DIR, 'baseline_failures.png')}")
# The model collapses each failure count to a single predicted value
# (it's a line), while actual grades spread widely at each level.
# This "flattening" reveals how much variance failures alone cannot explain.
'''
The model captures the overall negative trend between failures and grades, 
but it fails to explain the large variation within each group. 
Students with the same number of failures can have very different outcomes, 
which shows that additional features are needed for a useful model.
'''


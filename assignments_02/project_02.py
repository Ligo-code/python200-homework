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

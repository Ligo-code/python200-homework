import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

# =============================================================================
# --- Pandas ---
# =============================================================================

# Pandas Q1
# Create a DataFrame and print basic information about it
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print("=== Pandas Q1 ===")
print("First 3 rows:")
print(df.head(3))
print(f"Shape: {df.shape}")
print(f"Num Rows: {len(df)}")
print(f"Num Cols: {len(df.columns)}")
print("Data types:")
print(df.dtypes)

# Pandas Q2
# Filter rows: only students who passed AND have grade > 80
# The & operator combines two boolean conditions element-wise
print("\n=== Pandas Q2 ===")
passed_and_high = df[(df["passed"] == True) & (df["grade"] > 80)]
print("Students who passed and have grade > 80:")
print(passed_and_high)

# Pandas Q3
# Add a new column by applying a scalar operation to an existing column
# Pandas applies the +5 to every row automatically (vectorized operation)
print("\n=== Pandas Q3 ===")
df["grade_curved"] = df["grade"] + 5
print("DataFrame with curved grades:")
print(df)

# Pandas Q4
# The .str accessor lets us apply string methods to an entire column at once
# Without it, we'd have to loop over every row manually
print("\n=== Pandas Q4 ===")
df["name_upper"] = df["name"].str.upper()
print("Name and name_upper columns:")
print(df[["name", "name_upper"]])

# Pandas Q5
# groupby splits the DataFrame into groups based on a column value,
# then we apply an aggregation function (mean) to each group
print("\n=== Pandas Q5 ===")
city_mean_grade = df.groupby("city")["grade"].mean()
print("Mean grade per city:")
print(city_mean_grade)

# Pandas Q6
# .replace() substitutes one value for another throughout the column
print("\n=== Pandas Q6 ===")
df["city"] = df["city"].replace("Austin", "Houston")
print("Name and city after replacing Austin with Houston:")
print(df[["name", "city"]])

# Pandas Q7
# sort_values with ascending=False gives us descending order
# .head(3) returns just the top 3 rows
print("\n=== Pandas Q7 ===")
top3 = df.sort_values("grade", ascending=False).head(3)
print("Top 3 students by grade (descending):")
print(top3)

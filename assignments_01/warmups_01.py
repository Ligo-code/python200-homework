import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns
from statistics import mode


# =============================================================================
# --- Pandas ---
# =============================================================================
# Note: all Pandas questions (Q1-Q7) share the same DataFrame `df`

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
passed_and_high = df[(df["passed"]) & (df["grade"] > 80)]
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

# =============================================================================
# --- NumPy ---
# =============================================================================

# NumPy Q1
# np.array() creates an array from a Python list
# shape: dimensions as a tuple, dtype: data type, ndim: number of dimensions
print("\n=== NumPy Q1 ===")
arr1d = np.array([10, 20, 30, 40, 50])
print(f"Array: {arr1d}")
print(f"Shape: {arr1d.shape}")
print(f"Dtype: {arr1d.dtype}")
print(f"Ndim: {arr1d.ndim}")

# NumPy Q2
# A 2D array is like a matrix: shape gives (rows, cols)
# size is the total number of elements: rows * cols
print("\n=== NumPy Q2 ===")
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"Array:\n{arr}")
print(f"Shape: {arr.shape}")
print(f"Size (total elements): {arr.size}")

# NumPy Q3
# Slicing syntax: arr[row_start:row_end, col_start:col_end]
# :2 means "from index 0 up to (not including) index 2"
print("\n=== NumPy Q3 ===")
top_left = arr[:2, :2]
print(f"Top-left 2x2 block:\n{top_left}")

# NumPy Q4
# np.zeros() and np.ones() fill arrays with 0s and 1s respectively
# The argument is a tuple describing the shape: (rows, cols)
print("\n=== NumPy Q4 ===")
zeros = np.zeros((3, 4))
ones = np.ones((2, 5))
print(f"3x4 array of zeros:\n{zeros}")
print(f"2x5 array of ones:\n{ones}")

# NumPy Q5
# np.arange(start, stop, step) — like Python's range() but returns a NumPy array
# stop is exclusive, so arange(0, 50, 5) gives [0, 5, 10, ..., 45]
print("\n=== NumPy Q5 ===")
arange_arr = np.arange(0, 50, 5)
print(f"Array: {arange_arr}")
print(f"Shape: {arange_arr.shape}")
print(f"Mean: {arange_arr.mean()}")
print(f"Sum: {arange_arr.sum()}")
print(f"Std: {arange_arr.std()}")

# NumPy Q6
# np.random.normal(mean, std, size) draws random samples from a normal distribution
# With enough samples the computed mean and std should be close to the target values
print("\n=== NumPy Q6 ===")
np.random.seed(42)
random_arr = np.random.normal(0, 1, 200)
print(f"Mean of 200 random normal values: {random_arr.mean():.4f}  (target: 0)")
print(f"Std  of 200 random normal values: {random_arr.std():.4f}  (target: 1)")


# =============================================================================
# --- Matplotlib ---
# =============================================================================

# Matplotlib Q1
# plt.figure() starts a new empty figure so plots don't overlap each other
# plt.plot() draws a line through the given x, y points
print("\n=== Matplotlib Q1 ===")
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.figure()
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q2
# plt.bar() takes category labels and their corresponding heights
print("\n=== Matplotlib Q2 ===")
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

plt.figure()
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

# Matplotlib Q3
# plt.scatter() plots individual points (no connecting line)
# label= assigns a name to each dataset so plt.legend() can display it
print("\n=== Matplotlib Q3 ===")
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.figure()
plt.scatter(x1, y1, label="Data1")
plt.scatter(x2, y2, label="Data2")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q4
# plt.subplots(rows, cols) returns a figure and an array of Axes objects
# ax[0] and ax[1] are the individual subplot panels (left and right)
# On subplots we use ax.set_title() instead of plt.title()
# plt.tight_layout() automatically adjusts spacing so subplots don't overlap
print("\n=== Matplotlib Q4 ===")
fig, ax = plt.subplots(1, 2)

ax[0].plot(x, y)
ax[0].set_title("Line Plot")

ax[1].bar(subjects, scores)
ax[1].set_title("Bar Plot")

plt.tight_layout()
plt.show()

# =============================================================================
# --- Descriptive Stats ---
# =============================================================================

# Descriptive Stats Q1
# np.mean, np.median, np.var, np.std all accept plain Python lists as well as arrays
# variance measures average squared distance from the mean;
# std is its square root, expressed in the same units as the original data
print("\n=== Descriptive Stats Q1 ===")
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f"Mean:     {np.mean(data)}")
print(f"Median:   {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Std:      {np.std(data)}")

# Descriptive Stats Q2
# plt.figure() ensures the histogram is drawn on a fresh canvas, not on top of a previous plot
# np.random.seed makes the generated data identical on every run (reproducibility)
print("\n=== Descriptive Stats Q2 ===")
np.random.seed(42)
data = np.random.normal(65, 10, 500)
plt.figure()
plt.hist(data, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Q3
# A boxplot shows: median (center line), IQR (box), whiskers, and outliers (dots beyond whiskers)
# Passing a list of lists draws one box per group
print("\n=== Descriptive Stats Q3 ===")
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.figure()
plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.show()

# Descriptive Stats Q4
# The exponential distribution is right-skewed: most values are small,
# but a long tail of large values pulls the mean upward away from the bulk of the data.
# For skewed distributions, median is a better measure of central tendency
# because it is not affected by extreme values the way the mean is.
# The normal distribution is symmetric, so mean and median are roughly equal —
# either is an appropriate measure of central tendency.
print("\n=== Descriptive Stats Q4 ===")
np.random.seed(42)
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.figure()
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.show()

# Descriptive Stats Q5
# statistics.mode() from the standard library returns the single most common value.
# mode is the same for both datasets (12 appears twice), but mean differs greatly.
# Why? The outlier 150 in data2 inflates the mean because the mean sums all values.
# The median only looks at the middle position, so extreme values do not affect it.
print("\n=== Descriptive Stats Q5 ===")

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print("Data1 mean:", np.mean(data1), "median:", np.median(data1), "mode:", mode(data1))
print("Data2 mean:", np.mean(data2), "median:", np.median(data2), "mode:", mode(data2))

# data2 mean is skewed by the outlier 150 — it pulls the average up to 40,
# while the median stays at 12, which better represents the typical value


# =============================================================================
# --- Hypothesis Testing ---
# =============================================================================

# Hypothesis Q1
# ttest_ind tests whether two independent groups have the same mean
# A negative t-statistic means group_a mean is lower than group_b mean
print("\n=== Hypothesis Q1 ===")
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_val:.4f}")

# Hypothesis Q2
# alpha = 0.05 is the standard significance threshold:
# if p < 0.05 we say the result is statistically significant (less than 5% chance it's random)
print("\n=== Hypothesis Q2 ===")
if p_val < 0.05:
    print("Result is statistically significant (p < 0.05)")
else:
    print("Result is not statistically significant (p >= 0.05)")

# Hypothesis Q3
# ttest_rel is for paired data — the same subjects measured twice (before/after)
# unlike ttest_ind which assumes the two groups are independent
print("\n=== Hypothesis Q3 ===")
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat3, p_val3 = stats.ttest_rel(before, after)
print(f"Paired t-statistic: {t_stat3:.4f}")
print(f"p-value:            {p_val3:.4f}")

# Hypothesis Q4
# ttest_1samp tests whether a sample mean differs from a known reference value (popmean)
print("\n=== Hypothesis Q4 ===")
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat4, p_val4 = stats.ttest_1samp(scores, 70)
print(f"One-sample t-statistic: {t_stat4:.4f}")
print(f"p-value:                {p_val4:.4f}")

# Hypothesis Q5
# alternative='less' tests the one-sided hypothesis: group_a mean < group_b mean
# this gives a smaller p-value than the two-sided test when the direction is correct
print("\n=== Hypothesis Q5 ===")
one_tail_p = stats.ttest_ind(group_a, group_b, alternative='less').pvalue
print(f"One-tailed p-value (group_a < group_b): {one_tail_p:.4f}")

# Hypothesis Q6
# Plain-language conclusion for Q1: mention direction and whether it's due to chance
print("\n=== Hypothesis Q6 ===")
print(
    f"Group B scores (mean={np.mean(group_b):.1f}) were significantly higher than "
    f"Group A scores (mean={np.mean(group_a):.1f}). "
    f"The p-value of {p_val:.4f} is below 0.05, meaning this difference is very "
    f"unlikely to be due to chance."
)

# =============================================================================
# --- Correlation ---
# =============================================================================

# Correlation Q1
# np.corrcoef returns a 2x2 matrix: [0,0] and [1,1] are self-correlations (always 1.0),
# [0,1] and [1,0] are the actual correlation between x and y.
# y = 2x is a perfect linear relationship, so we expect correlation = 1.0
print("\n=== Correlation Q1 ===")
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr = np.corrcoef(x, y)
print("Correlation matrix:\n", corr)
print("Coefficient:", corr[0, 1])
# Expected: 1.0 — y is exactly 2*x, a perfect positive linear relationship.
# The result is 0.9999... instead of exactly 1.0 due to floating-point arithmetic:
# computers represent decimals approximately, so tiny rounding errors accumulate.

# Correlation Q2
# pearsonr returns (correlation_coefficient, p_value)
# p-value tells us whether the correlation is statistically significant
print("\n=== Correlation Q2 ===")
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]

r, p = pearsonr(x, y)
print(f"Pearson r: {r:.4f}")
print(f"p-value:   {p:.4f}")
# r ≈ -0.976: strong negative correlation — as x increases, y decreases almost perfectly.
# Not exactly -1.0 because y has slight noise (e.g. 7 and 8 are swapped around x=3,4).

# Correlation Q3
# df.corr() computes pairwise Pearson correlation for all numeric columns
# values range from -1 (perfect negative) to +1 (perfect positive)
print("\n=== Correlation Q3 ===")
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 72, 80],
    "age":    [25, 30, 22, 35, 28]
}
df_corr = pd.DataFrame(people)
print("Correlation matrix:\n", df_corr.corr())

# Correlation Q4
# plt.figure() starts a fresh canvas so this plot doesn't overlap the previous one
print("\n=== Correlation Q4 ===")
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.figure()
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Correlation Q5
# sns.heatmap visualizes a matrix as a color grid
# annot=True overlays the numeric values on each cell
print("\n=== Correlation Q5 ===")
plt.figure()
sns.heatmap(df_corr.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# =============================================================================
# --- Pipelines ---
# =============================================================================

# Pipeline Q1
# A pipeline is a chain of functions where each step transforms the data
# and passes the result to the next step — no special framework needed.

def create_series(arr):
    # Convert a NumPy array into a named pandas Series
    return pd.Series(arr, name="values")

def clean_data(series):
    # Remove NaN values so they don't affect statistical calculations
    return series.dropna()

def summarize_data(series):
    # Compute summary statistics and return them as a dictionary
    # series.mode()[0] gets the first (most common) mode value
    return {
        "mean":   series.mean(),
        "median": series.median(),
        "std":    series.std(),
        "mode":   series.mode()[0]
    }

def data_pipeline(arr):
    # Orchestrates the three steps: load → clean → summarize
    s = create_series(arr)
    s_clean = clean_data(s)
    return summarize_data(s_clean)

print("\n=== Pipeline Q1 ===")
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

result = data_pipeline(arr)

print("Pipeline results:")
for k, v in result.items():
    print(f"  {k}: {v}")


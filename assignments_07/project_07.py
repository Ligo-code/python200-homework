import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool


# ==================================================
# Setup
# ==================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

DATA_PATH = BASE_DIR.parent / "assignments_01" / "outputs" / "merged_happiness.csv"
PLOT_PATH = OUTPUTS_DIR / "happiness_by_region.png"

df = None

print("Environment loaded.")
print("Data path:", DATA_PATH)


# ==================================================
# Task 1 — Tool 1: load_happiness_data
# ==================================================

@tool
def load_happiness_data() -> dict:
    """
    Load the World Happiness dataset into memory.

    Returns:
        A dictionary containing the shape, column names, data path, and output plot path.
    """
    global df

    if not DATA_PATH.exists():
        return {"error": f"Data file not found at {DATA_PATH}"}

    df = pd.read_csv(DATA_PATH)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "data_path": str(DATA_PATH),
        "plot_path": str(PLOT_PATH),
    }


# ==================================================
# Task 1 — Tool 2: summarize_column
# ==================================================

@tool
def summarize_column(column: str) -> dict:
    """
    Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column: The name of the column to summarize.

    Returns:
        A dictionary with descriptive statistics for the selected column.
    """
    global df

    if df is None:
        return {"error": "No data is loaded. Please call load_happiness_data first."}

    if column not in df.columns:
        return {
            "error": f"Column '{column}' not found. Available columns: {df.columns.tolist()}"
        }

    return df[column].describe().to_dict()


# ==================================================
# Task 1 — Tool 3: compute_correlation
# ==================================================

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: The name of the first numeric column.
        col2: The name of the second numeric column.

    Returns:
        A dictionary containing the correlation coefficient and p-value.
    """
    global df

    if df is None:
        return {"error": "No data is loaded. Please call load_happiness_data first."}

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns not found in dataset."}

    try:
        pearson_r, p_value = pearsonr(df[col1], df[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(p_value), 4),
        }

    except Exception as e:
        return {"error": str(e)}


# ==================================================
# Task 1 — Tool 4: get_top_n_countries
# ==================================================

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """
    Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The numeric column used for ranking.
        year: The year to filter the dataset by.
        n: Number of top countries to return.

    Returns:
        A dictionary containing the ranked countries and values.
    """
    global df

    if df is None:
        return {"error": "No data is loaded. Please call load_happiness_data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    filtered_df = df[df["year"] == year]

    if filtered_df.empty:
        return {"error": f"No data found for year {year}."}

    top_rows = filtered_df.sort_values(by=column, ascending=False).head(n)

    result = []

    for _, row in top_rows.iterrows():
        result.append(
            {
                "country": row["country"],
                column: row[column],
            }
        )

    return {"results": result}


# ==================================================
# Task 2 — Build the Agent
# ==================================================

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini",
)

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.

Use the available tools for:
- loading data
- summarizing columns
- computing correlations
- ranking countries

Important rules — follow them exactly:
1. NEVER mock or generate fake data. Always use the tools to load real data.
2. When creating any plot, ALWAYS start with these exact lines FIRST:
       import matplotlib
       matplotlib.use('Agg')
       import matplotlib.pyplot as plt
3. For custom plots, call load_happiness_data() first, read the CSV from the
   returned data_path using pandas, then save the plot to the exact absolute
   path returned as plot_path. Never use a relative path like 'outputs/...'.

Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[
        load_happiness_data,
        summarize_column,
        compute_correlation,
        get_top_n_countries,
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=[
        "pandas",
        "matplotlib",
        "matplotlib.pyplot",
        "scipy.stats",
    ],
    max_steps=8,
)


# ==================================================
# Task 3 — Run Guided Queries
# ==================================================

if __name__ == "__main__":
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        (
            "Plot happiness_score over the years as a line chart, with one line per region. "
            "Use the data_path and plot_path from load_happiness_data. "
            "Read the real CSV with pandas, group by year and regional_indicator, "
            "and save the plot exactly to plot_path."
        ),
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    print("\nExpected plot location:")
    print(PLOT_PATH)

    # ==================================================
    # Task 4 — Your Own Questions
    # ==================================================

    my_query_1 = (
        "Which country improved its happiness_score the most between 2015 and 2023?"
    )
    print(f"\n--- My Query 1: {my_query_1} ---")
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    
    # Comment: This triggered code generation because the agent needed to compare
    # happiness_score changes between multiple years across countries.

    my_query_2 = (
       "Which region has the highest average happiness_score across all years?"
    )
    print(f"\n--- My Query 2: {my_query_2} ---")
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)

    # Comment: This triggered code generation because the agent needed to group
    # the dataset by region and calculate average happiness scores.

    # ==================================================
# Task 5 — Reflection
# ==================================================

# --- Reflection ---
#
# 1. In Query 3, the agent communicated statistical significance by returning
#    both the Pearson correlation coefficient and the p-value. It used the
#    p-value correctly because a p-value of 0.0 is well below the standard
#    significance threshold of 0.05, so the correlation was considered
#    statistically significant.
#
# 2. One response that surprised me was Query 5, where the agent successfully
#    generated custom pandas and matplotlib code to create a line chart grouped
#    by region. This was more capable than I expected because none of the
#    predefined tools directly supported multi-line plotting, so the agent had
#    to reason through the steps and write the analysis code itself.
#
# 3. One additional tool that would make this agent more useful would be a
#    custom plotting tool that accepts parameters such as x column, y column,
#    grouping column, chart type, and output filename. This would allow the
#    agent to create visualizations more reliably without needing to generate
#    raw Python code, and it would help answer questions like:
#    "Create a bar chart of average happiness_score by region for 2020."

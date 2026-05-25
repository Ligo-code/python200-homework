# World Happiness Agent

A mini-project built with `smolagents` and OpenAI's `CodeAgent` to explore the World Happiness dataset through natural language queries.

## Project Overview

This project demonstrates how an AI agent can combine predefined Python tools with dynamic code generation to analyze data conversationally.

The agent can:

- load the World Happiness dataset
- summarize dataset columns
- compute Pearson correlations with statistical significance
- rank countries by selected metrics
- generate custom visualizations when predefined tools are not sufficient

## Technologies Used

- Python
- Pandas
- Matplotlib
- SciPy
- smolagents
- OpenAI API
- dotenv

## Project Structure

```bash
assignments_07/
│
├── project_07.py
├── outputs/
│   └── happiness_by_region.png
```

## Implemented Tools

### load_happiness_data()

Loads the merged World Happiness dataset into a shared global DataFrame.

Returns:

- dataset shape
- column names
- data path
- plot output path

---

### summarize_column(column)

Returns descriptive statistics for a selected column.

Example:

```python
summarize_column("happiness_score")
```

---

### compute_correlation(col1, col2)

Computes:

- Pearson correlation coefficient
- p-value

Example:

```python
compute_correlation("gdp_per_capita", "happiness_score")
```

---

### get_top_n_countries(column, year, n)

Ranks countries by a selected metric for a given year.

Example:

```python
get_top_n_countries("happiness_score", 2020, 5)
```

## Example Queries

Guided queries:

- Load the happiness data and tell me its shape and column names
- Summarize the happiness_score column
- Compute correlation between GDP per capita and happiness score
- Show top 5 happiest countries in 2020
- Generate a regional happiness trend chart

Custom queries:

- Which country improved its happiness score the most between 2015 and 2023?
- Which region has the highest average happiness score across all years?

## Key Learning Outcomes

This project helped demonstrate the difference between:

**Tool-based agent behavior**
- safe
- predictable
- limited to predefined actions

and

**CodeAgent behavior**
- flexible
- capable of custom analysis
- able to generate Python code dynamically
- more prone to execution/runtime issues

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python assignments_07/project_07.py
```

## Output

Generated visualization:

```bash
assignments_07/outputs/happiness_by_region.png
```
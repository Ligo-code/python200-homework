import numpy as np
import pandas as pd
from prefect import flow, task


# Pipeline Q2
# Same logic as warmups_01.py Pipeline Q1, but each step is now a Prefect @task.
# @task tells Prefect to track this function: log its state, handle retries, etc.
# @flow is the entry point that orchestrates all tasks in order.

@task
def create_series(arr):
    """Convert a NumPy array into a named pandas Series."""
    return pd.Series(arr, name="values")


@task
def clean_data(series):
    """Remove NaN values so they don't affect statistical calculations."""
    return series.dropna()


@task
def summarize_data(series):
    """Compute summary statistics and return them as a dictionary."""
    # series.mode()[0] gets the first (most common) mode value
    return {
        "mean":   series.mean(),
        "median": series.median(),
        "std":    series.std(),
        "mode":   series.mode()[0]
    }


@flow
def pipeline_flow():
    """Orchestrate the three tasks: load → clean → summarize."""
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan,
                    18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)

    print("\nPipeline summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return summary


if __name__ == "__main__":
    pipeline_flow()


# =============================================================================
# Reflection questions
# =============================================================================

# Q: Why might Prefect be more overhead than it is worth here?
# This pipeline is tiny: three functions, a handful of numbers, runs in under a second.
# Prefect adds imports, decorators, an orchestration engine, and background state tracking
# — all of which is unnecessary complexity for something this simple.
# Plain Python functions are easier to read, debug, and run with no extra setup.

# Q: Describe realistic scenarios where Prefect could still be useful,
#    even if the pipeline logic itself stays simple.
# - Scheduled runs: re-running the same pipeline every night on fresh data
#   without manually triggering it.
# - Automatic retries: if a step reads from an API or S3 bucket and the network
#   blips, Prefect can retry automatically instead of failing silently.
# - Visibility: when multiple pipelines run in production, the Prefect dashboard
#   shows which flows succeeded, failed, or are still running — hard to replicate
#   with plain print() statements.
# - Parameterization: running the same flow with different inputs (different date
#   ranges, different file paths) without changing the code.

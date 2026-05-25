# ==================================================
# Lesson 02: Tool Definitions and the ReAct Loop
# ==================================================

import os
import json
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI


# ==================================================
# Setup
# ==================================================

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("OpenAI client created.")


# ==================================================
# Q1
# ==================================================

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "Temperature in Celsius",
                }
            },
            "required": ["celsius"],
        },
    },
}


print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

"""
Output:
0°C is 32.0°F
100°C is 212.0°F
-40°C is -40.0°F
"""


# ==================================================
# Q2
# ==================================================

"""
Prediction:

Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call?
No.

Why?
Because this version of the agent only has one available tool: get_current_time.
The query asks for a temperature conversion, not the current time.

How many API calls will be made?
One API call.

If the model does not request a tool, the first response is already the final answer.
"""


def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


time_tool_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Returns the current local time as a string.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


tools = [time_tool_schema]
print("Tools list defined with one tool: get_current_time")


def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt."""

    SYSTEM_PROMPT = """
    You are a simple assistant that can tell the current time and convert Celsius to Fahrenheit.
    Use tools when they are helpful.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    print("First response received from model...")
    print(first_response)

    first_message = first_response.choices[0].message

    messages.append(
        {
            "role": "assistant",
            "content": first_message.content,
            "tool_calls": first_message.tool_calls,
        }
    )

    if first_message.tool_calls:
        print("Agentic mode engaged...")

        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name

            if function_name == "get_current_time":
                tool_result = get_current_time()

            elif function_name == "celsius_to_fahrenheit":
                args = json.loads(tool_call.function.arguments)
                tool_result = celsius_to_fahrenheit(args["celsius"])

            else:
                tool_result = f"Error: unknown tool {function_name}."

            print("Tool called:", function_name)
            print("Tool result:", tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )

        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ""

    print("No tools needed....")
    return first_message.content or ""


answer_with_agent = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(answer_with_agent)

# Was my prediction correct?
# Yes. The model answered directly without calling the tool because
# get_current_time is unrelated to temperature conversion.


# ==================================================
# Q3
# ==================================================

tools = [
    time_tool_schema,
    celsius_to_fahrenheit_schema,
]

print("Tools list updated with two tools: get_current_time and celsius_to_fahrenheit")

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)

# A tool was called because the user explicitly requested a temperature conversion,
# and the celsius_to_fahrenheit tool was available to perform that calculation.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)

# No tool was called because this was a conceptual knowledge question.
# The model already knows the boiling point of water from its training data.

# ==================================================
# Lesson 03: Multi-Tool Agent
# Q4
# ==================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr


RESOURCES_DIR = Path(__file__).parent / "resources"


class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    def list_csv_files(self):
        """List available CSV files in resources/."""
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """Load a CSV file from resources/ and make it the active dataset."""
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """Return column names for the currently loaded CSV."""
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """Return basic summary stats for one or more columns."""
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """Simple summary for a single column using pandas.describe()."""
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """Plot from the active CSV."""
        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        if x == y:
            x = None

        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            
            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.savefig("assignments_07/outputs/my_plot.png")
        

        return f"Plotted {y} vs {x} as a {plot_type}."

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error

        missing = [col for col in [col1, col2] if col not in self.df.columns]
        if missing:
            return {"error": f"These columns are not in the data: {missing}"}

        pearson_r, p_value = pearsonr(self.df[col1], self.df[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(p_value), 4),
        }


print("Class defined")

csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation coefficient and p-value between two columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "First column name.",
                    },
                    "col2": {
                        "type": "string",
                        "description": "Second column name.",
                    },
                },
                "required": ["col1", "col2"],
            },
        },
    },
]

# ==================================================
# Q5
# ==================================================

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one ReAct agent loop using tool calling.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        content = json.dumps(result, default=str) if not isinstance(result, str) else result

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    for loop_idx in range(max_tool_rounds):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        assistant_entry = {
            "role": "assistant",
            "content": msg.content,
        }

        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]

        messages.append(assistant_entry)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)

            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}

            messages.append(observe_tool_result(tool_call.id, result))

    return "I hit the tool-round limit. Try a simpler request."


SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)


messages = [{"role": "system", "content": SYSTEM_PROMPT}]

result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
)

print(result)

'''
Output:

ACT: list_csv_files({})
ACT: load_csv({'filename': 'bike_commute.csv'})
ACT: compute_correlation({'col1': 'avg_traffic_density', 'col2': 'avg_speed_kmh'})
The correlation between average traffic density and average speed (km/h) is approximately -0.53. 
This indicates a moderate negative correlation, meaning that as traffic density increases, average speed tends to decrease. 
The p-value is 0.0, showing this correlation is statistically significant.
'''

# ==================================================
# Q6
# ==================================================

# In the ReAct loop:
# system = instructions that define the agent's behavior and rules.
# user = the user's request or question.
# assistant = the model's reasoning step, either a final answer or a request to call tools.
# tool = the observation step, where Python returns the result of the requested tool call.

print(json.dumps(messages, indent=2, default=str))

# ==================================================
# Lesson 04: smolagents
# Q7
# ==================================================

from smolagents import tool


@tool
def compute_correlation_tool(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation between two columns in the currently loaded CSV file.

    Args:
        col1: The name of the first numeric column.
        col2: The name of the second numeric column.
    """
    return csv_backend.compute_correlation(col1, col2)


print(compute_correlation_tool.description)

# Comparison:
# In Q4 we manually wrote a full JSON schema with function name,
# description, parameters, and required fields.
# smolagents generates this automatically from the function signature
# and docstring.

# To generate a good description, the developer must provide:
# 1. clear function name
# 2. meaningful parameter names
# 3. helpful docstring describing what the tool does

# ==================================================
# Q8
# ==================================================

from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel


model = OpenAIServerModel(
    model_id="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)


@tool
def load_csv_tool(filename: str) -> dict:
    """
    Load a CSV file from the resources directory.

    Args:
        filename: The CSV filename to load.
    """
    return csv_backend.load_csv(filename)


@tool
def plot_data_tool(y: str, x: str = None, plot_type: str = "line") -> str:
    """
    Plot data from the currently loaded CSV file.

    Args:
        y: Column name for y-axis.
        x: Optional column name for x-axis.
        plot_type: Plot type, either scatter or line.
    """
    return csv_backend.plot_data(y, x, plot_type)


TOOLS = [
    load_csv_tool,
    plot_data_tool,
    compute_correlation_tool,
]


tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

response_tool = tool_agent.run(prompt)
print("ToolCallingAgent response:", response_tool)

response_code = code_agent.run(
    prompt,
    additional_args={"csv_manager": csv_backend},
)
print("CodeAgent response:", response_code)

"""
Comparison:

ToolCallingAgent used the predefined tools:
- load_csv_tool
- plot_data_tool

It created a scatter plot, but the dots were the default matplotlib color, not green.
This happened because plot_data_tool does not have a color parameter, so the agent could not pass "green" into the tool.

CodeAgent attempted to generate and execute Python code dynamically.
However, it failed because matplotlib tried to create a GUI figure from a worker thread on macOS.
This shows both the strength and the risk of CodeAgent:
it is more flexible because it can write code, but it can also run into execution/environment errors.

A ToolCallingAgent is better for controlled, predictable workflows.
A CodeAgent is more useful when the task requires custom logic that was not built into the predefined tools.
"""

# ==================================================
# Q9
# ==================================================

"""
A ToolCallingAgent would be a better choice for a task like loading a CSV file,
summarizing columns, or computing a predefined statistic such as correlation.

This is a good fit for a tool-based approach because the task has clear,
limited actions and safe predefined tools. The agent only needs to choose
which tool to call and what arguments to pass.

One meaningful risk of using a CodeAgent is that it can generate and execute
unexpected or unsafe code. For example, it may try to import unauthorized
libraries, access attributes that do not exist, modify files, or fail because
of environment-specific issues. A ToolCallingAgent does not have this same
level of risk because it can only call the tools that the developer explicitly
provided.
"""
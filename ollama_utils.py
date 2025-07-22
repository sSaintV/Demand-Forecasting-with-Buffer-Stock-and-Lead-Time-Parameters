import streamlit as st
import pandas as pd
import re
from datetime import timedelta
import ollama
import io

@st.cache_data
def load_data_poc2(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df["WEEK"] = pd.to_datetime(df["WEEK"])
    df = df.sort_values(["PRODUCT", "WEEK"])
    return df

def prepare_weekly_data_poc2(df, level="PRODUCT"):
    weekly_df = df.groupby(["WEEK", level])["UNITS"].sum().reset_index()
    all_weeks = pd.date_range(start=df["WEEK"].min(), end=df["WEEK"].max(), freq="W-MON")
    levels = df[level].unique()

    filled_data = []
    for l in levels:
        temp = weekly_df[weekly_df[level] == l].set_index("WEEK").reindex(all_weeks, fill_value=0)
        temp[level] = l
        temp["WEEK"] = temp.index
        filled_data.append(temp.reset_index(drop=True))

    return pd.concat(filled_data)

def generate_prompt_ollama(weeks, units, label, future_weeks):
    history = "\n".join([f"{w.date()}: {u} units" for w, u in zip(weeks, units)])
    date_cols = " | ".join([w.strftime("%d-%m-%Y") for w in future_weeks])
    header = f"| Product ID | {date_cols} |"
    sep = "|" + "-----------|" * (1 + len(future_weeks))
    row = f"| {label} | {' | '.join(['[unit]']*len(future_weeks))} |"
    return f"""You are a demand forecasting expert. Analyze the following weekly units for product ID {label}:

{history}

Please forecast the next {len(future_weeks)} weeks of units using a robust time series model (such as ARIMA, Prophet, or Exponential Smoothing), while also incorporating ML Algorithms (such as XGBoost or Extra Trees) for better accuracy. Consider seasonality, trends, and any anomalies in the data.



Return the forecast as a markdown table in this format, with the forecasted week dates as columns and a single row for the forecasted units for product {label}:

{header}
{sep}
{row}

Replace [unit] with the forecasted value for each week. After the table, provide a list of insights gained from the forecasted units, such as expected trends, seasonality effects, or any anomalies detected in the data. Also provide a paragraph of actionable insights based on the forecasted units, such as potential stock adjustments or marketing strategies.
"""

def run_ollama_forecast(prompt, model="llama3"):
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

@st.cache_data(show_spinner=False)
def cached_llm_response(prompt, model="llama3"):
    return run_ollama_forecast(prompt, model)

def parse_forecast_ollama(output, historical_units=None):
    lines = output.split("\n")
    table_lines = []
    in_table = False
    for line in lines:
        if line.strip().startswith("|") and "|" in line[1:]:
            if set(line.replace("|", "").replace("-", "").strip()) == set():
                continue
            table_lines.append(line)
            in_table = True
        elif in_table and not line.strip().startswith("|"):
            break
    if table_lines:
        clean_table_lines = [l for l in table_lines if not re.match(r"^\|[\s\-|]+\|$", l)]
        table_str = "\n".join([l.strip().strip("|").replace(" | ", ",") for l in clean_table_lines])
        df = pd.read_csv(io.StringIO(table_str), header=0)
        if "Units(DF)" in df.columns:
            df = df.drop(columns=["Units(DF)"])
        forecast = []
        for val in df.iloc[0, 1:]:
            try:
                forecast.append(int(round(float(val))))
            except Exception:
                forecast.append(0)
        if historical_units is not None and len(historical_units) > 0:
            min_unit = int(min(historical_units))
            max_unit = int(max(historical_units))
            forecast = [min(max(unit, min_unit), max_unit) for unit in forecast]
        header = "| Product ID | " + " | ".join(df.columns[1:]) + " |"
        sep = "|" + "-----------|" * (1 + len(df.columns[1:]))
        rounded_units = [str(unit) for unit in forecast]
        row = f"| {df.iloc[0,0]} | " + " | ".join(rounded_units) + " |"
        markdown_table = header + "\n" + sep + "\n" + row
        table_end_idx = lines.index(table_lines[-1])
        insights = "\n".join(lines[table_end_idx+1:]).strip()
        return forecast, markdown_table, insights
    else:
        numbers = re.findall(r"\d+\.?\d*", output)
        forecast = [int(round(float(x))) for x in numbers[:12]]
        if historical_units is not None and len(historical_units) > 0:
            min_unit = int(min(historical_units))
            max_unit = int(max(historical_units))
            forecast = [min(max(unit, min_unit), max_unit) for unit in forecast]
        if len(forecast) < 12:
            forecast += [forecast[-1]] * (12 - len(forecast))
        markdown_table = ""
        insights = "\n".join(output.split("\n")[1:])
        return forecast, markdown_table, insights
    
def build_chat_prompt_ollama(user_input, sub_df, future_weeks, forecast):
    hist = "\n".join([f"{w.date()}: {int(u)} units" for w, u in zip(sub_df["WEEK"][-8:], sub_df["UNITS"][-8:])])
    fut = "\n".join([f"{w.date()}: {int(u)} units (forecast)" for w, u in zip(future_weeks, forecast)])
    context = (
        "You are a demand forecasting assistant. Here is the recent historical data:\n"
        f"{hist}\n\n"
        "And here is the forecast for the next 12 weeks:\n"
        f"{fut}\n\n"
        f"User question: {user_input}\n"
        "Please answer using the data above."
    )
    return context



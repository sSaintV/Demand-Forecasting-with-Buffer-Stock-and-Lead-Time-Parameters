import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import timedelta
import ollama
import io

st.set_page_config(layout="wide")

# --- Helper Functions ---
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['WEEK'] = pd.to_datetime(df['WEEK'])
    df = df.sort_values(['PRODUCT', 'WEEK'])
    return df

def prepare_weekly_data(df, level='PRODUCT'):
    weekly_df = df.groupby(['WEEK', level])['UNITS'].sum().reset_index()
    all_weeks = pd.date_range(start=df['WEEK'].min(), end=df['WEEK'].max(), freq='W-MON')
    levels = df[level].unique()

    filled_data = []
    for l in levels:
        temp = weekly_df[weekly_df[level] == l].set_index('WEEK').reindex(all_weeks, fill_value=0)
        temp[level] = l
        temp['WEEK'] = temp.index
        filled_data.append(temp.reset_index(drop=True))

    return pd.concat(filled_data)

def generate_prompt(weeks, units, label, future_weeks):
    history = "\n".join([f"{w.date()}: {u} units" for w, u in zip(weeks, units)])
    date_cols = " | ".join([w.strftime("%d-%m-%Y") for w in future_weeks])
    header = f"| Product ID | Units(DF) | {date_cols} |"
    sep = "|" + "-----------|" * (2 + len(future_weeks))
    row = f"| {label} |           | {' | '.join(['[unit]']*len(future_weeks))} |"
    return f"""You are a demand forecasting expert. Analyze the following weekly units for product ID {label}:

{history}

Please forecast the next {len(future_weeks)} weeks of units using a robust time series model (such as ARIMA, Prophet, or Exponential Smoothing), considering seasonality, trends, and any anomalies.

Return the forecast as a markdown table in this format, with the forecasted week dates as columns and a single row for the forecasted units for product {label}:

{header}
{sep}
{row}

Replace [unit] with the forecasted value for each week. After the table, provide a list of insights gained from the forecasted unit sales and specify which forecasting method was used.
"""

def run_ollama_forecast(prompt, model="llama3"):
    # Use the ollama Python package instead of subprocess
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    # The response is a dict with a 'message' key containing another dict with 'content'
    return response['message']['content']

@st.cache_data(show_spinner=False)
def cached_llm_response(prompt, model="llama3"):
    return run_ollama_forecast(prompt, model)

def parse_forecast(output):
    # Find the markdown table in the output
    lines = output.split('\n')
    table_lines = []
    in_table = False
    for line in lines:
        if line.strip().startswith('|') and '|' in line[1:]:
            table_lines.append(line)
            in_table = True
        elif in_table and not line.strip().startswith('|'):
            break
    if not table_lines:
        # fallback to old method if no table found
        numbers = re.findall(r'\d+', output)
        forecast = list(map(int, numbers[:12]))
        if len(forecast) < 12:
            forecast += [forecast[-1]] * (12 - len(forecast))
        insights = "\n".join(output.split("\n")[1:])
        return forecast, insights

    # Convert markdown table to CSV-like string
    table_str = '\n'.join([l.strip().strip('|').replace(' | ', ',') for l in table_lines])
    df = pd.read_csv(io.StringIO(table_str), header=0)
    # The forecasted units are in the first row after the first two columns
    # Ensure forecasted units are whole numbers (integers, rounded up if needed)
    forecast = df.iloc[0, 2:].apply(lambda x: int(round(float(x)))).tolist()
    # Insights: everything after the table
    table_end_idx = lines.index(table_lines[-1])
    insights = "\n".join(lines[table_end_idx+1:]).strip()
    return forecast, insights

def build_chat_prompt(user_input, sub_df, future_weeks, forecast):
    hist = "\n".join([f"{w.date()}: {int(u)} units" for w, u in zip(sub_df['WEEK'][-8:], sub_df['UNITS'][-8:])])
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


@st.cache_data(show_spinner=False)
def cached_llm_response(prompt, model="llama3"):
    return run_ollama_forecast(prompt, model)

# --- Streamlit UI ---
st.title("📈 Electronics Demand Forecasting with Ollama")
st.markdown("Forecast weekly unit sales using LLMs running locally via Ollama.")

uploaded_file = st.file_uploader("Upload your 3-year electronics weekly sales CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    level = "PRODUCT"
    weekly_df = prepare_weekly_data(df, level=level)

    mode = st.selectbox("Select forecasting type:", ["Single Product", "Multiple Products", "All Products"])

    if mode == "Single Product":
        selected_group = st.selectbox(f"Select {level} ID to forecast:", sorted(df[level].unique()))
        sub_df = weekly_df[weekly_df[level] == selected_group]
        recent_data = sub_df[-52:]
        future_weeks = [weekly_df['WEEK'].max() + timedelta(weeks=i) for i in range(1, 13)]
        prompt = generate_prompt(recent_data['WEEK'], recent_data['UNITS'], selected_group, future_weeks)

        # Only run the forecast if the button is pressed or forecast_result is not in session_state
        if st.button("Run Forecast with Ollama") or "forecast_result" in st.session_state:
            # Only run the LLM if forecast_result is not in session_state
            if "forecast_result" not in st.session_state:
                with st.spinner("Running local LLM forecasting..."):
                    output = run_ollama_forecast(prompt)
                    forecast, insights = parse_forecast(output)
                    last_week = weekly_df['WEEK'].max()
                    future_weeks = [last_week + timedelta(weeks=i) for i in range(1, 13)]
                    st.session_state.forecast_result = {
                        "output": output,
                        "forecast": forecast,
                        "insights": insights,
                        "last_week": last_week,
                        "future_weeks": future_weeks,
                        "selected_group": selected_group,
                        "sub_df": sub_df
                    }

            # Always use the forecast from session_state
            result = st.session_state.forecast_result
            forecast = result["forecast"]
            insights = result["insights"]
            last_week = result["last_week"]
            future_weeks = result["future_weeks"]
            selected_group = result["selected_group"]
            sub_df = result["sub_df"]

            col_chart_hist, col_chart_forecast, col_chat = st.columns([1, 1, 1])

            # Historical Data Chart
            with col_chart_hist:
                st.subheader("Historical Data")
                fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                ax_hist.plot(sub_df['WEEK'], sub_df['UNITS'], label='Historical', color='blue')
                ax_hist.set_xlabel("Date")
                ax_hist.set_ylabel("Units")
                ax_hist.set_title(f"Product {selected_group} - Historical Units")
                ax_hist.legend()
                st.pyplot(fig_hist)

            # Forecasted Data Chart
            with col_chart_forecast:
                st.subheader("Forecasted Data")
                fig_forecast, ax_forecast = plt.subplots(figsize=(6, 4))
                ax_forecast.plot(result["future_weeks"], result["forecast"], label='Forecast', color='orange', linestyle='--', marker='o')
                ax_forecast.set_xlabel("Date")
                ax_forecast.set_ylabel("Units")
                ax_forecast.set_title(f"Product {selected_group} - Forecasted Units")
                ax_forecast.legend()
                st.pyplot(fig_forecast)

            # Chat remains unchanged
            with col_chat:
                st.subheader("💬 Chat with Llama 3")
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                for msg in st.session_state.chat_history:
                    st.chat_message(msg["role"]).write(msg["content"])

                user_input = st.chat_input("Ask the LLM about demand, trends, or anything else...")

                if user_input:
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.chat_message("user").write(user_input)

                    with st.spinner("Llama 3 is thinking..."):
                        chat_prompt = build_chat_prompt(
                            user_input,
                            sub_df,
                            future_weeks,
                            forecast
                        )
                        response = cached_llm_response(chat_prompt)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.chat_message("assistant").write(response)

            st.markdown("---")
            st.subheader("Forecast Insights")
            st.markdown(insights)
            #st.markdown("#### Sample Prompt Preview")
            #st.code(prompt.strip(), language='text')

    elif mode == "Multiple Products":
        selected_products = st.multiselect("Select one or more Product IDs to forecast:", sorted(df['PRODUCT'].unique()))
        combine_chart = st.checkbox("Combine forecasts into one chart", value=True)

        if st.button("Run Forecast for Selected Products") and selected_products:
            results = []
            with st.spinner("Running forecasts for selected product IDs..."):
                last_week = weekly_df['WEEK'].max()
                for pid in selected_products:
                    sub_df = weekly_df[weekly_df['PRODUCT'] == pid]
                    recent_data = sub_df[-52:]
                    prompt = generate_prompt(recent_data['WEEK'], recent_data['UNITS'], pid)
                    output = run_ollama_forecast(prompt)
                    forecast, insights = parse_forecast(output)
                    future_weeks = [last_week + timedelta(weeks=i) for i in range(1, 13)]
                    results.append((pid, sub_df, future_weeks, forecast, insights))

            if combine_chart:
                fig, ax = plt.subplots(figsize=(12, 6))
                for pid, sub_df, future_weeks, forecast, _ in results:
                    ax.plot(sub_df['WEEK'], sub_df['UNITS'], label=f'{pid} - Historical')
                    ax.plot(future_weeks, forecast, label=f'{pid} - Forecast', linestyle='--')
                ax.set_title("Combined Forecasts")
                ax.legend()
                st.pyplot(fig)

            for pid, _, _, _, insights in results:
                st.subheader(f"Forecast Insights - Product {pid}")
                #st.markdown(insights)
                #st.markdown("---")

    elif mode == "All Products":
        combine_chart = st.checkbox("Combine forecasts into one chart", value=True)

        if st.button("Run Forecast for All Products"):
            results = []
            with st.spinner("Running forecasts for all product IDs..."):
                last_week = weekly_df['WEEK'].max()
                for pid in sorted(df['PRODUCT'].unique()):
                    sub_df = weekly_df[weekly_df['PRODUCT'] == pid]
                    recent_data = sub_df[-52:]
                    prompt = generate_prompt(recent_data['WEEK'], recent_data['UNITS'], pid)
                    output = run_ollama_forecast(prompt)
                    forecast, insights = parse_forecast(output)
                    future_weeks = [last_week + timedelta(weeks=i) for i in range(1, 13)]
                    results.append((pid, sub_df, future_weeks, forecast, insights))

            if combine_chart:
                fig, ax = plt.subplots(figsize=(12, 6))
                for pid, sub_df, future_weeks, forecast, _ in results:
                    ax.plot(sub_df['WEEK'], sub_df['UNITS'], label=f'{pid} - Historical')
                    ax.plot(future_weeks, forecast, label=f'{pid} - Forecast', linestyle='--')
                ax.set_title("Combined Forecasts for All Products")
                ax.legend()
                st.pyplot(fig)

            for pid, _, _, _, insights in results:
                st.subheader(f"Forecast Insights - Product {pid}")
                #st.markdown(insights)
                #st.markdown("---")
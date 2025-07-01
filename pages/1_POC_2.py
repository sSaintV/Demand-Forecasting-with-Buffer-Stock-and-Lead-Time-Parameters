import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import timedelta
import ollama

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
    # Use the ollama Python package instead of subprocess
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    # The response is a dict with a 'message' key containing another dict with 'content'
    return response['message']['content']

@st.cache_data(show_spinner=False)
def cached_llm_response(prompt, model="llama3"):
    return run_ollama_forecast(prompt, model)

def parse_forecast(output, historical_units=None):
    import io
    # Extract markdown table if present and ensure units are rounded and clipped to historical min/max
    lines = output.split('\n')
    table_lines = []
    in_table = False
    for line in lines:
        if line.strip().startswith('|') and '|' in line[1:]:
            # Skip separator lines (those that are only dashes and pipes)
            if set(line.replace('|', '').replace('-', '').strip()) == set():
                continue
            table_lines.append(line)
            in_table = True
        elif in_table and not line.strip().startswith('|'):
            break
    if table_lines:
        # Remove separator lines (those with only dashes and pipes)
        clean_table_lines = [l for l in table_lines if not re.match(r'^\|[\s\-|]+\|$', l)]
        # Remove the Units(DF) column (second column) from header and row
        # Convert markdown table to CSV-like string, then drop the second column
        table_str = '\n'.join([l.strip().strip('|').replace(' | ', ',') for l in clean_table_lines])
        df = pd.read_csv(io.StringIO(table_str), header=0)
        if "Units(DF)" in df.columns:
            df = df.drop(columns=["Units(DF)"])
        # The forecasted units are in the first row after the first column
        forecast = []
        for val in df.iloc[0, 1:]:
            try:
                forecast.append(int(round(float(val))))
            except Exception:
                forecast.append(0)
        # Clip forecast to historical min/max if provided
        if historical_units is not None and len(historical_units) > 0:
            min_unit = int(min(historical_units))
            max_unit = int(max(historical_units))
            forecast = [min(max(unit, min_unit), max_unit) for unit in forecast]
        # Build the markdown table in the requested format (without Units(DF))
        header = "| Product ID | " + " | ".join(df.columns[1:]) + " |"
        sep = "|" + "-----------|" * (1 + len(df.columns[1:]))
        rounded_units = [str(unit) for unit in forecast]
        row = f"| {df.iloc[0,0]} | " + " | ".join(rounded_units) + " |"
        markdown_table = header + "\n" + sep + "\n" + row
        table_end_idx = lines.index(table_lines[-1])
        insights = "\n".join(lines[table_end_idx+1:]).strip()
        return forecast, markdown_table, insights
    else:
        # fallback to old method if no table found
        numbers = re.findall(r'\d+\.?\d*', output)
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
                    forecast, markdown_table, insights = parse_forecast(output, historical_units=sub_df['UNITS'].values)
                    last_week = weekly_df['WEEK'].max()
                    future_weeks = [last_week + timedelta(weeks=i) for i in range(1, 13)]
                    st.session_state.forecast_result = {
                        "output": output,
                        "forecast": forecast,
                        "markdown_table": markdown_table,
                        "insights": insights,
                        "last_week": last_week,
                        "future_weeks": future_weeks,
                        "selected_group": selected_group,
                        "sub_df": sub_df
                    }

            # Always use the forecast from session_state
            result = st.session_state.forecast_result
            forecast = result["forecast"]
            markdown_table = result["markdown_table"]
            insights = result["insights"]
            last_week = result["last_week"]
            future_weeks = result["future_weeks"]
            selected_group = result["selected_group"]
            sub_df = result["sub_df"]

            col_chart_hist, col_chart_forecast = st.columns([1, 1])

            # Historical Data Chart
            with col_chart_hist:
                st.subheader("Historical Data")
                fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                ax_hist.plot(sub_df['WEEK'], sub_df['UNITS'], label='Historical', color='blue')
                ax_hist.set_ylabel("Units")
                ax_hist.set_title(f"Product {selected_group} - Historical Units")

                # Set x-axis by quarter, showing all quarters from first to last record
                weeks = pd.Series(sub_df['WEEK'].unique())
                if not weeks.empty:
                    # Find the first and last week
                    start = weeks.min()
                    end = weeks.max()
                    # Generate quarter start dates from start to end
                    quarter_starts = pd.date_range(
                        start=start.to_period('Q').start_time, 
                        end=end.to_period('Q').end_time, 
                        freq='QS'
                    )
                    # Set ticks at quarter starts, label as Q{quarter} {year}
                    ax_hist.set_xticks(quarter_starts)
                    ax_hist.set_xticklabels([f"Q{q.quarter} {q.year}" for q in quarter_starts], rotation=45, ha='right')
                    ax_hist.set_xlabel("Quarter")
                else:
                    ax_hist.set_xlabel("Quarter")

                # --- Get y-limits for matching ---
                y_min = min(sub_df['UNITS'].min(), min(result["forecast"]))
                y_max = max(sub_df['UNITS'].max(), max(result["forecast"]))
                ax_hist.set_ylim(y_min, y_max)

                ax_hist.legend()
                st.pyplot(fig_hist)

            # Forecasted Data Chart
            with col_chart_forecast:
                st.subheader("Forecasted Data")
                fig_forecast, ax_forecast = plt.subplots(figsize=(6, 4))
                ax_forecast.plot(result["future_weeks"], result["forecast"], label='Forecast', color='orange', linestyle='--', marker='o')
                # Set x-axis by month
                future_weeks = pd.Series(result["future_weeks"])
                month_starts = future_weeks[future_weeks.dt.is_month_start]
                if not month_starts.empty:
                    ax_forecast.set_xticks(month_starts)
                    ax_forecast.set_xticklabels([q.strftime("%b %Y") for q in month_starts], rotation=45, ha='right')
                else:
                    fallback_ticks = future_weeks[::4]
                    ax_forecast.set_xticks(fallback_ticks)
                    ax_forecast.set_xticklabels([q.strftime("%b %Y") for q in fallback_ticks], rotation=45, ha='right')
                ax_forecast.set_xlabel("Month")
                ax_forecast.set_ylabel("Units")
                ax_forecast.set_title(f"Product {selected_group} - Forecasted Units")
                # --- Set y-limits to match historical chart ---
                ax_forecast.set_ylim(y_min, y_max)
                ax_forecast.legend()
                st.pyplot(fig_forecast)

            st.markdown("---")
            st.subheader("Forecast Insights")
            st.markdown("**Forecast Table:**")
            st.markdown(markdown_table)
            st.markdown("**Insights:**")
            st.markdown(insights)

            st.markdown("---")
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

    elif mode == "Multiple Products":
        selected_products = st.multiselect("Select one or more Product IDs to forecast:", sorted(df['PRODUCT'].unique()))
        # Store results in session_state to persist after button click
        if "multiple_products_results" not in st.session_state:
            st.session_state["multiple_products_results"] = None

        if st.button("Run Forecast for Selected Products") and selected_products:
            results = []
            with st.spinner("Running forecasts for selected product IDs..."):
                last_week = weekly_df['WEEK'].max()
                for pid in sorted(selected_products):  # sort for lowest ID first
                    sub_df = weekly_df[weekly_df['PRODUCT'] == pid]
                    recent_data = sub_df[-52:]
                    prompt = generate_prompt(recent_data['WEEK'], recent_data['UNITS'], pid, [last_week + timedelta(weeks=i) for i in range(1, 13)])
                    output = run_ollama_forecast(prompt)
                    forecast, markdown_table, insights = parse_forecast(output, historical_units=sub_df['UNITS'].values)
                    future_weeks = [last_week + timedelta(weeks=i) for i in range(1, 13)]
                    results.append((pid, sub_df, future_weeks, forecast, markdown_table, insights))
            st.session_state["multiple_products_results"] = results

        # Use results from session_state if available
        results = st.session_state.get("multiple_products_results", [])

        if results:
            col_chart_hist, col_chart_forecast = st.columns([1, 1])

            with col_chart_hist:
                combined_hist = weekly_df[weekly_df['PRODUCT'].isin(selected_products)]
                fig_hist, ax_hist = plt.subplots(figsize=(12, 5))
                for pid in sorted(selected_products):
                    sub_df = combined_hist[combined_hist['PRODUCT'] == pid]
                    ax_hist.plot(sub_df['WEEK'], sub_df['UNITS'], label=f'{pid} - Historical')
                ax_hist.set_ylabel("Units")
                ax_hist.set_title("Historical Units (All Selected Products)")
                all_weeks = pd.Series(combined_hist['WEEK'].unique())
                if not all_weeks.empty:
                    start = all_weeks.min()
                    end = all_weeks.max()
                    quarter_starts = pd.date_range(
                        start=start.to_period('Q').start_time,
                        end=end.to_period('Q').end_time,
                        freq='QS'
                    )
                    ax_hist.set_xticks(quarter_starts)
                    ax_hist.set_xticklabels([f"Q{q.quarter} {q.year}" for q in quarter_starts], rotation=45, ha='right')
                    ax_hist.set_xlabel("Quarter")
                else:
                    ax_hist.set_xlabel("Quarter")
                all_units = combined_hist['UNITS'].values
                all_forecasts = [unit for _, _, _, forecast, _, _ in results for unit in forecast]
                y_min = min(all_units.min(), min(all_forecasts))
                y_max = max(all_units.max(), max(all_forecasts))
                ax_hist.set_ylim(y_min, y_max)
                ax_hist.legend()
                st.pyplot(fig_hist)

            with col_chart_forecast:
                fig_forecast, ax_forecast = plt.subplots(figsize=(12, 5))
                for pid, _, future_weeks, forecast, _, _ in results:
                    ax_forecast.plot(future_weeks, forecast, label=f'{pid} - Forecast', linestyle='--', marker='o')
                all_future_weeks = pd.Series([w for _, _, future_weeks, _, _, _ in results for w in future_weeks])
                month_starts = all_future_weeks[all_future_weeks.dt.is_month_start].drop_duplicates()
                if not month_starts.empty:
                    ax_forecast.set_xticks(month_starts)
                    ax_forecast.set_xticklabels([q.strftime("%b %Y") for q in month_starts], rotation=45, ha='right')
                else:
                    fallback_ticks = all_future_weeks[::4].drop_duplicates()
                    ax_forecast.set_xticks(fallback_ticks)
                    ax_forecast.set_xticklabels([q.strftime("%b %Y") for q in fallback_ticks], rotation=45, ha='right')
                ax_forecast.set_xlabel("Month")
                ax_forecast.set_ylabel("Units")
                ax_forecast.set_title("Forecasted Units (All Selected Products)")
                ax_forecast.set_ylim(y_min, y_max)
                ax_forecast.legend()
                st.pyplot(fig_forecast)

            for pid, _, _, _, markdown_table, insights in results:
                st.markdown(f"---\n### Product {pid} Forecast Insights")
                if markdown_table:
                    st.markdown("**Forecast Table:**")
                    st.markdown(markdown_table)
                st.markdown("**Insights:**")
                st.markdown(insights)
        else:
            st.info("No forecasts to display. Please select products and run the forecast.")

        # --- Chat Section (persists after forecast) ---
        st.markdown("---")
        st.subheader("💬 Chat with Llama 3 about All Selected Products")
        chat_key = "chat_history_multiple_products"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        for msg in st.session_state[chat_key]:
            st.chat_message(msg["role"]).write(msg["content"])

        user_input = st.chat_input("Ask about any of the selected products...", key="chat_input_multiple_products")

        if user_input and results:
            st.session_state[chat_key].append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            combined_histories = []
            combined_forecasts = []
            for pid, sub_df, future_weeks, forecast, _, _ in results:
                hist = "\n".join([f"{w.date()}: {int(u)} units" for w, u in zip(sub_df['WEEK'][-8:], sub_df['UNITS'][-8:])])
                fut = "\n".join([f"{w.date()}: {int(u)} units (forecast)" for w, u in zip(future_weeks, forecast)])
                combined_histories.append(f"Product {pid}:\n{hist}")
                combined_forecasts.append(f"Product {pid}:\n{fut}")
            chat_prompt = (
                "You are a demand forecasting assistant. Here is the recent historical data for the selected products:\n"
                + "\n\n".join(combined_histories)
                + "\n\nAnd here are the forecasts for the next 12 weeks:\n"
                + "\n\n".join(combined_forecasts)
                + f"\n\nUser question: {user_input}\nPlease answer using the data above."
            )

            with st.spinner("Llama 3 is thinking..."):
                response = cached_llm_response(chat_prompt)
            st.session_state[chat_key].append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

    elif mode == "All Products":
        combine_chart = st.checkbox("Combine forecasts into one chart", value=True)

        if st.button("Run Forecast for All Products"):
            results = []
            with st.spinner("Running forecasts for all product IDs..."):
                last_week = weekly_df['WEEK'].max()
                for pid in sorted(df['PRODUCT'].unique()):
                    sub_df = weekly_df[weekly_df['PRODUCT'] == pid]
                    recent_data = sub_df[-52:]
                    prompt = generate_prompt(recent_data['WEEK'], recent_data['UNITS'], pid, [last_week + timedelta(weeks=i) for i in range(1, 13)])
                    output = run_ollama_forecast(prompt)
                    forecast, markdown_table, insights = parse_forecast(output, historical_units=sub_df['UNITS'].values)
                    future_weeks = [last_week + timedelta(weeks=i) for i in range(1, 13)]
                    results.append((pid, sub_df, future_weeks, forecast, markdown_table, insights))
            st.session_state["all_products_results"] = results

        # Use results from session_state if available
        results = st.session_state.get("all_products_results", [])

        if results:
            # --- Combined Historical Data Chart ---
            st.subheader("Historical Data (All Products)")
            fig_hist, ax_hist = plt.subplots(figsize=(12, 5))
            for pid, sub_df, _, _, _, _ in results:
                ax_hist.plot(sub_df['WEEK'], sub_df['UNITS'], label=f'{pid} - Historical')
            ax_hist.set_ylabel("Units")
            ax_hist.set_title("Historical Units (All Products)")
            all_weeks = pd.Series(weekly_df['WEEK'].unique())
            if not all_weeks.empty:
                start = all_weeks.min()
                end = all_weeks.max()
                quarter_starts = pd.date_range(
                    start=start.to_period('Q').start_time,
                    end=end.to_period('Q').end_time,
                    freq='QS'
                )
                ax_hist.set_xticks(quarter_starts)
                ax_hist.set_xticklabels([f"Q{q.quarter} {q.year}" for q in quarter_starts], rotation=45, ha='right')
                ax_hist.set_xlabel("Quarter")
            else:
                ax_hist.set_xlabel("Quarter")
            all_units = weekly_df['UNITS'].values
            all_forecasts = [unit for _, _, _, forecast, _, _ in results for unit in forecast]
            y_min = min(all_units.min(), min(all_forecasts))
            y_max = max(all_units.max(), max(all_forecasts))
            ax_hist.set_ylim(y_min, y_max)
            ax_hist.legend()
            st.pyplot(fig_hist)

            # --- Combined Forecasted Data Chart ---
            st.subheader("Forecasted Data (All Products)")
            fig_forecast, ax_forecast = plt.subplots(figsize=(12, 5))
            for pid, _, future_weeks, forecast, _, _ in results:
                ax_forecast.plot(future_weeks, forecast, label=f'{pid} - Forecast', linestyle='--', marker='o')
            all_future_weeks = pd.Series([w for _, _, future_weeks, _, _, _ in results for w in future_weeks])
            month_starts = all_future_weeks[all_future_weeks.dt.is_month_start].drop_duplicates()
            if not month_starts.empty:
                ax_forecast.set_xticks(month_starts)
                ax_forecast.set_xticklabels([q.strftime("%b %Y") for q in month_starts], rotation=45, ha='right')
            else:
                fallback_ticks = all_future_weeks[::4].drop_duplicates()
                ax_forecast.set_xticks(fallback_ticks)
                ax_forecast.set_xticklabels([q.strftime("%b %Y") for q in fallback_ticks], rotation=45, ha='right')
            ax_forecast.set_xlabel("Month")
            ax_forecast.set_ylabel("Units")
            ax_forecast.set_title("Forecasted Units (All Products)")
            ax_forecast.set_ylim(y_min, y_max)
            ax_forecast.legend()
            st.pyplot(fig_forecast)

            # --- Per-product Insights ---
            for pid, sub_df, future_weeks, forecast, markdown_table, insights in results:
                st.markdown(f"---\n### Product {pid}")
                st.subheader("Forecast Insights")
                if markdown_table:
                    st.markdown("**Forecast Table:**")
                    st.markdown(markdown_table)
                st.markdown("**Insights:**")
                st.markdown(insights)
        else:
            st.info("No forecasts to display. Please run the forecast.")

        # --- Chat Section for All Products ---
        st.markdown("---")
        st.subheader("💬 Chat with Llama 3 about All Products")
        chat_key = "chat_history_all_products"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        for msg in st.session_state[chat_key]:
            st.chat_message(msg["role"]).write(msg["content"])

        user_input = st.chat_input("Ask about any of the products...", key="chat_input_all_products")

        if user_input and results:
            st.session_state[chat_key].append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            combined_histories = []
            combined_forecasts = []
            for pid, sub_df, future_weeks, forecast, _, _ in results:
                hist = "\n".join([f"{w.date()}: {int(u)} units" for w, u in zip(sub_df['WEEK'][-8:], sub_df['UNITS'][-8:])])
                fut = "\n".join([f"{w.date()}: {int(u)} units (forecast)" for w, u in zip(future_weeks, forecast)])
                combined_histories.append(f"Product {pid}:\n{hist}")
                combined_forecasts.append(f"Product {pid}:\n{fut}")
            chat_prompt = (
                "You are a demand forecasting assistant. Here is the recent historical data for all products:\n"
                + "\n\n".join(combined_histories)
                + "\n\nAnd here are the forecasts for the next 12 weeks:\n"
                + "\n\n".join(combined_forecasts)
                + f"\n\nUser question: {user_input}\nPlease answer using the data above."
            )

            with st.spinner("Llama 3 is thinking..."):
                response = cached_llm_response(chat_prompt)
            st.session_state[chat_key].append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
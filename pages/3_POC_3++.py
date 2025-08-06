import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import base64
from ollama_utils import generate_prompt_ollama, run_ollama_forecast, parse_forecast_ollama, build_chat_prompt_ollama, cached_llm_response

def consolidate_lead_time(df, lead_time_col="Lead Time (weeks)", demand_col="New Demand Forecast", group_cols=["Product"]):
    """
    Applies lead time consolidation to the demand column for each product.
    Sets the first (Lead Time - 1) weeks to zero, and sums those values into the Lead Time week.
    Leaves subsequent weeks unchanged.
    """
    df = df.copy()
    if lead_time_col not in df.columns:
        df[lead_time_col] = 0
    # If not grouped by product, just use all rows
    if group_cols and all(col in df.columns for col in group_cols):
        grouped = df.groupby(group_cols)
        for name, group in grouped:
            lt = int(group[lead_time_col].iloc[0])
            if lt > 1 and len(group) >= lt:
                idxs = group.index.tolist()
                # Zero out first (lt-1) weeks
                df.loc[idxs[:lt-1], demand_col] = 0
                # Sum first (lt) weeks into the Lead Time week
                df.loc[idxs[lt-1], demand_col] = group[demand_col].iloc[:lt].sum()
    else:
        lt = int(df[lead_time_col].iloc[0]) if lead_time_col in df.columns else 0
        if lt > 1 and len(df) >= lt:
            idxs = df.index.tolist()
            df.loc[idxs[:lt-1], demand_col] = 0
            df.loc[idxs[lt-1], demand_col] = df[demand_col].iloc[:lt].sum()
    return df

# Set page configuration
st.set_page_config(
    page_title="Demand Forecasting with AI Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Simulator Reset Functionality ---
def save_simulator_original(display_adjusted):
    """
    Save the original, unedited simulator table for reset functionality.
    """
    # Always ensure baseline columns exist for reset
    sim_base = display_adjusted.copy()
    if "New Demand Forecast" not in sim_base.columns:
        sim_base["New Demand Forecast"] = sim_base["Base Demand Forecast"]
    if "New Inventory Forecast" not in sim_base.columns:
        sim_base["New Inventory Forecast"] = sim_base["Inventory"]
    st.session_state["sim_base_original"] = sim_base

def reset_simulator_to_original():
    """
    Restore the simulator table to its original values and clear all edits.
    """
    if "sim_base_original" in st.session_state:
        sim_base = st.session_state["sim_base_original"].copy()
        # Always ensure columns exist after reset
        if "New Demand Forecast" not in sim_base.columns:
            sim_base["New Demand Forecast"] = sim_base["Base Demand Forecast"]
        if "New Inventory Forecast" not in sim_base.columns:
            sim_base["New Inventory Forecast"] = sim_base["Inventory"]
        st.session_state["sim_base_df"] = sim_base
    st.session_state["edit_changes"] = []
    st.success("Simulator table reset to original values.")
    st.rerun()

def generate_prompt_ollama(weeks, base_forecast, product_id, future_weeks, actuals=None, historical_weeks=None, historical_units=None):
    """
    Advanced prompt engineering for Llama 3:
    Generates high-quality, product-specific AI insights for each SKU and forecast period.
    Focuses exclusively on Base forecast values, but includes historical data for seasonality and accuracy metrics.
    """
    # Format forecast weeks
    weeks = pd.to_datetime(weeks)
    if hasattr(weeks, "dt"):
        week_strs = weeks.dt.strftime("%Y-%m-%d")
    else:
        week_strs = pd.Series(weeks).dt.strftime("%Y-%m-%d")
    df_forecast = pd.DataFrame({
        "Week": week_strs,
        "Base Forecast": [int(round(x)) for x in base_forecast]
    })

    # Format historical data if provided
    historical_table_md = ""
    if historical_weeks is not None and historical_units is not None:
        hist_weeks = pd.to_datetime(historical_weeks)
        if hasattr(hist_weeks, "dt"):
            hist_week_strs = hist_weeks.dt.strftime("%Y-%m-%d")
        else:
            hist_week_strs = pd.Series(hist_weeks).dt.strftime("%Y-%m-%d")
        df_hist = pd.DataFrame({
            "Week": hist_week_strs,
            "Actuals": [int(round(x)) for x in historical_units]
        })
        historical_table_md = "\n#### Historical Actuals\n" + df_hist.to_markdown(index=False) + "\n"

    # If actuals are provided for forecast period, add them for accuracy metrics
    if actuals is not None:
        df_forecast["Actuals"] = [int(round(x)) for x in actuals]

    table_md = df_forecast.to_markdown(index=False)

    prompt = f"""
You are an advanced demand forecasting and inventory optimization AI. Analyze the following Base forecast data for product SKU '{product_id}':

#### Forecast Table
{table_md}
{historical_table_md}

For this product and forecast period, provide detailed, actionable insights in the following structured format:

---

### 1. **Seasonality Detection**
- Identify and describe any repeating seasonal patterns, peaks, or troughs in the forecasted demand, using both the historical actuals and forecast data.
- Specify which weeks or periods show significant seasonality or anomalies.

### 2. **Inventory Recommendations**
- Recommend optimal inventory levels for each forecasted week, considering demand variability and potential stockouts.
- Highlight any weeks where inventory risk is high and suggest mitigation strategies.

### 3. **Forecast Accuracy Metrics**
- If actuals are provided, calculate and report forecast accuracy metrics (e.g., MAE, MAPE, RMSE) for the available periods.
- Comment on the reliability of the forecast for this SKU.

### 4. **Outlier Identification**
- Detect and list any outlier weeks in the forecasted demand (unusually high or low values).
- Suggest possible causes for these outliers.

### 5. **Trend Analysis**
- Analyze the overall demand trend (increasing, decreasing, stable, or volatile) using both historical and forecast data.
- Quantify the trend (e.g., average weekly change, % growth/decline).

### 6. **Demand Pattern Insights**
- Summarize key demand patterns for this SKU.
- Provide actionable business recommendations for promotions, supply chain, or pricing based on the forecast.

---

**Important Instructions:**
- Use only the Base forecast values and historical actuals provided above as your data source.
- Do not reference or compare to any other forecast scenarios or products.
- Segment your analysis clearly for this product and forecast period.
- Respond in markdown format, using bullet points and bolded section headers as shown above.
"""
    return prompt

def generate_multi_product_prompt(forecast_df, actuals_df=None):
    """
    Generates a single prompt that segments insights by product (SKU) and forecast period.
    Each product gets a dedicated section with its own table and analysis.
    """
    prompt_sections = []
    for product in forecast_df["Product"].unique():
        prod_df = forecast_df[forecast_df["Product"] == product]
        weeks = prod_df["Week"]
        base_forecast = prod_df["Base Demand Forecast"] if "Base Demand Forecast" in prod_df.columns else prod_df["UNITS"]
        actuals = None
        if actuals_df is not None and product in actuals_df["Product"].unique():
            actuals = actuals_df[actuals_df["Product"] == product]["UNITS"]
        section = generate_prompt_ollama(
            weeks=weeks,
            base_forecast=base_forecast,
            product_id=product,
            future_weeks=weeks.tolist(),
            actuals=actuals
        )
        prompt_sections.append(f"## Product: {product}\n\n{section}")
    return "\n\n---\n\n".join(prompt_sections)

# --- Robust error handling for CSV parsing in main app ---
def safe_read_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if "WEEK" not in df.columns or "UNITS" not in df.columns or "SALES" not in df.columns:
            raise ValueError("Missing required columns: WEEK, UNITS, SALES")
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV file contains the required columns: WEEK, UNITS, SALES")
        return None


def recalculate_inventory(sim_base, demand_col="New Demand Forecast", inventory_col="New Inventory Forecast"):
    """
    Recalculates inventory for each product after any change.
    Preserves any manually changed inventory values for a week,
    and resumes normal calculation for subsequent weeks using the manual value as the new baseline.
    """
    sim_base = sim_base.copy()
    sim_base["Week"] = pd.to_datetime(sim_base["Week"])
    sim_base = sim_base.sort_values(["Product", "Week"]).reset_index(drop=True)

    # Identify all manual inventory overrides: (Product, Week) -> value
    manual_inventory = {}
    edit_changes = st.session_state.get("edit_changes", [])
    for edit in edit_changes:
        if (edit.get("Value Type") == "Inventory" or edit.get("Field Type") == "Inventory"):
            product = str(edit.get("Product", ""))
            week = pd.to_datetime(edit.get("Demand Forecast Week"))
            manual_inventory[(product, week)] = edit.get("New Unit")

    for product in sim_base["Product"].unique():
        product_mask = sim_base["Product"] == product
        product_df = sim_base[product_mask].sort_values("Week")
        weeks = product_df["Week"].tolist()
        for i, week in enumerate(weeks):
            idx = product_df.index[i]
            if i == 0:
                continue  # First week inventory stays as is (could be user-edited)
            prev_idx = product_df.index[i - 1]
            prev_inventory = sim_base.loc[prev_idx, inventory_col]
            curr_demand = sim_base.loc[idx, demand_col]
            manual_key = (str(product), week)
            if manual_key in manual_inventory:
                sim_base.loc[idx, inventory_col] = manual_inventory[manual_key]
            else:
                sim_base.loc[idx, inventory_col] = prev_inventory - curr_demand
    return sim_base


# --- UNIVERSAL PRODUCT FILTERING ---
def filter_by_product(df, product_filter, product_col="PRODUCT"):
    if product_filter is not None and product_col in df.columns:
        return df[df[product_col].astype(str).isin(product_filter)]
    return df


def calculate_inventory_matrix(df, product_col="Product", week_col="Week", demand_col="Base Demand Forecast", start_inventory_col="Inventory"):
    """
    Calculates inventory for each product and week using:
    Inventory_this_week = Inventory_last_week - Demand_this_week
    Returns a DataFrame with products as rows and weeks as columns.
    """
    df = df.copy()
    df[product_col] = df[product_col].astype(str)
    df[week_col] = pd.to_datetime(df[week_col])
    df = df.sort_values([product_col, week_col])

    products = df[product_col].unique()
    weeks = sorted(df[week_col].unique())
    week_labels = [str(w.date()) for w in weeks]

    result = {}
    for product in products:
        product_df = df[df[product_col] == product].set_index(week_col)
        inventory = []
        for i, week in enumerate(weeks):
            if week in product_df.index:
                demand = product_df.loc[week, demand_col]
                if i == 0:
                    start_inv = product_df.loc[week, start_inventory_col]
                    inventory.append(start_inv)
                else:
                    prev_inv = inventory[-1]
                    inventory.append(prev_inv - demand)
            else:
                inventory.append(np.nan)
        result[product] = inventory

    inventory_matrix = pd.DataFrame(result, index=week_labels).transpose()
    inventory_matrix.index.name = "Product"
    return inventory_matrix

def edit_forecast_data(adjusted_forecast, forecast_range, df=None):
    st.sidebar.header("🛠️ Simulator")
    edit_field = st.sidebar.selectbox(
        "Select Field to Edit",
        options=["Demand Forecast", "Inventory"],
        index=0,
        key="simulator_field_select"
    )
    if edit_field == "Demand Forecast":
        field = "ADJUSTED_FORECAST"
        table_type = "Unit"
    else:
        field = "INVENTORY_FORECAST"
        table_type = "Inventory"

    data_source = adjusted_forecast
    key_prefix = "forecast"

    forecast_weeks = data_source[
        (data_source["WEEK"] >= forecast_range[0]) & (data_source["WEEK"] <= forecast_range[1])
    ]["WEEK"].dt.strftime("%Y-%m-%d").unique()

    if len(forecast_weeks) == 0:
        st.sidebar.info("No forecast weeks in the selected range.")
        return

    week_val = st.sidebar.selectbox(
        "Select Forecast Week",
        forecast_weeks,
        key=f"{key_prefix}_week_selectbox"
    )

    product_val = None
    if "PRODUCT" in data_source.columns:
        product_filter = st.session_state.get("active_product_filter", None)
        available_products = sorted(data_source["PRODUCT"].astype(str).unique())
        if product_filter:
            available_products = [p for p in available_products if p in product_filter]
        product_val = st.sidebar.selectbox(
            "Select Product",
            available_products,
            key=f"{key_prefix}_product_selectbox"
        )

    mask = (data_source["WEEK"].dt.strftime("%Y-%m-%d") == week_val)
    if product_val is not None:
        mask &= (data_source["PRODUCT"].astype(str) == product_val)

    sim_base = st.session_state.get("sim_base_df", None)
    current_val = None
    if edit_field == "Inventory":
        if sim_base is not None:
            sim_mask = (sim_base["Week"].astype(str) == week_val)
            if product_val is not None:
                sim_mask &= (sim_base["Product"].astype(str) == product_val)
            if sim_mask.any():
                current_val = sim_base.loc[sim_mask, "New Inventory Forecast"].iloc[0]
        if current_val is None and mask.any():
            current_val = data_source.loc[mask, "INVENTORY_FORECAST"].iloc[0]
    else:
        if sim_base is not None:
            sim_mask = (sim_base["Week"].astype(str) == week_val)
            if product_val is not None:
                sim_mask &= (sim_base["Product"].astype(str) == product_val)
            if sim_mask.any():
                current_val = sim_base.loc[sim_mask, "New Demand Forecast"].iloc[0]
        if current_val is None and mask.any():
            current_val = data_source.loc[mask, field].iloc[0]

    new_val = st.sidebar.number_input(
        f"Set new value for {edit_field} on {week_val}" + (f" (Product: {product_val})" if product_val else ""),
        value=int(round(current_val)) if current_val is not None else 0,
        step=1,
        format="%d",
        key=f"{key_prefix}_num_input"
    )

    if "edit_changes" not in st.session_state:
        st.session_state["edit_changes"] = []

    if st.sidebar.button("Apply Change", key=f"{key_prefix}_apply_btn"):
        # Remove any previous edit for this week/product/field to avoid duplicates
        st.session_state["edit_changes"] = [
            e for e in st.session_state["edit_changes"]
            if not (
                e.get("Value Type") == table_type and
                e.get("Demand Forecast Week") == week_val and
                (e.get("Product", "") == (product_val if product_val else ""))
            )
        ]
        st.session_state["edit_changes"].append({
            "Value Type": table_type,
            "Field Type": table_type,
            "Demand Forecast Week": week_val,
            "Product": product_val if product_val else "",
            "Original Unit": int(current_val) if current_val is not None else None,
            "New Unit": new_val
        })
        st.rerun()


# --- POC_1 DemandForecaster Class ---
class DemandForecaster:
    def __init__(self):
        self.data = None
        self.forecast_data = None
        self.model = None
        self.scaler = None
        
    def preprocess_data(self, df):
        try:
            df["WEEK"] = pd.to_datetime(df["WEEK"])
        except Exception as e:
            st.error(
                "Error converting WEEK column to datetime. "
                "Please ensure WEEK column contains valid dates."
            )
            return None
            
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        categorical_columns = df.select_dtypes(include=["object"]).columns
        df[categorical_columns] = df[categorical_columns].fillna("Unknown")
        
        df = df.sort_values("WEEK")
        
        return df

    def create_forecast_features(self, df, target_col):
        group_cols = ["WEEK"]
        if "PRODUCT" in df.columns:
            group_cols.append("PRODUCT")

        agg_dict = {
            target_col: "sum",
            "BASE_PRICE": "mean",
            "PRICE": "mean",
        }
        
        if "DISPLAY" in df.columns:
            agg_dict["DISPLAY"] = "max"
        if "INVENTORY" in df.columns:
            agg_dict["INVENTORY"] = "sum"
        if "VISITS" in df.columns:
            agg_dict["VISITS"] = "sum"
        if "FEATURE" in df.columns:
            agg_dict["FEATURE"] = "max"

        weekly_data = df.groupby(group_cols).agg(agg_dict).reset_index()

        for lag in [1, 2, 4, 8]:
            weekly_data[f"{target_col}_lag_{lag}"] = weekly_data[target_col].shift(lag)

        for window in [4, 8, 12]:
            weekly_data[f"{target_col}_rolling_{window}"] = weekly_data[target_col].rolling(window=window).mean()

        weekly_data["week_number"] = weekly_data["WEEK"].dt.isocalendar().week
        weekly_data["month"] = weekly_data["WEEK"].dt.month
        weekly_data["quarter"] = weekly_data["WEEK"].dt.quarter

        weekly_data["price_ratio"] = weekly_data["PRICE"] / weekly_data["BASE_PRICE"]
        weekly_data["price_ratio"] = weekly_data["price_ratio"].fillna(1)

        return weekly_data
    
    def train_forecast_model(self, df, target_col, forecast_periods=12, model_type="Random Forest"):
        feature_columns = [col for col in df.columns if col not in ["WEEK", target_col] and not col.startswith(target_col)]
        
        clean_df = df.dropna()
        
        if len(clean_df) < 10:
            st.error("Not enough data points for forecasting. Need at least 10 complete records.")
            return None, None
        
        X = clean_df[feature_columns]
        y = clean_df[target_col]
        
        split_idx = int(len(clean_df) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        if model_type == "Linear Regression":
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            self.model = LinearRegression()
            self.model.fit(X_train_scaled, y_train)
        else:
            self.model = RandomForestRegressor(
                n_estimators=375,
                random_state=42,
                max_depth=10,
                min_samples_split=10
                )
            self.model.fit(X_train, y_train)

        forecast_dates = pd.date_range(start=df["WEEK"].max() + pd.Timedelta(weeks=1), 
                                     periods=forecast_periods, freq="W")
        
        last_row = clean_df.iloc[-1:].copy()
        forecasts = []

        if "PRODUCT" in clean_df.columns:
            product_value = clean_df["PRODUCT"].iloc[-1]
        else:
            product_value = None
        
        for i, date in enumerate(forecast_dates):
            future_row = last_row.copy()
            future_row["WEEK"] = date
            future_row["week_number"] = date.isocalendar()[1]
            future_row["month"] = date.month
            future_row["quarter"] = date.quarter

            if product_value is not None:
                future_row["PRODUCT"] = product_value
            
            if i == 0:
                for lag in [1, 2, 4, 8]:
                    if f"{target_col}_lag_{lag}" in future_row.columns:
                        future_row[f"{target_col}_lag_{lag}"] = clean_df[target_col].iloc[-lag] if len(clean_df) >= lag else clean_df[target_col].iloc[-1]
            
            X_future = future_row[feature_columns]
            if model_type == "Linear Regression":
                X_future = self.scaler.transform(X_future)
            pred = self.model.predict(X_future)[0]
            
            forecasts.append({
                "WEEK": date,
                "FORECAST": max(0, pred),
                "TYPE": "Original",
                "PRODUCT": product_value if product_value is not None else np.nan
            })
        
        forecast_df = pd.DataFrame(forecasts)
        
        if model_type == "Linear Regression":
            historical_pred = self.model.predict(self.scaler.transform(X))
        else:
            historical_pred = self.model.predict(X)
        historical_df = pd.DataFrame({
            "WEEK": clean_df["WEEK"],
            "ACTUAL": clean_df[target_col],
            "FORECAST": historical_pred,
            "TYPE": "Historical"
        })
        return forecast_df, historical_df
    
    def apply_supply_chain_adjustments(self, forecast_df, lead_time_weeks, safety_pct):
        adjusted_forecast = forecast_df.copy()
        
        if lead_time_weeks > 0:
            adjusted_forecast["WEEK"] = adjusted_forecast["WEEK"] + pd.Timedelta(weeks=lead_time_weeks)
        
        adjusted_forecast["SAFETY_PCT_ADJ"] = adjusted_forecast["FORECAST"] * (safety_pct / 100)
        adjusted_forecast["ADJUSTED_FORECAST"] = adjusted_forecast["FORECAST"] + adjusted_forecast["SAFETY_PCT_ADJ"]

        adjusted_forecast["ADJUSTED_FORECAST"] = adjusted_forecast["ADJUSTED_FORECAST"].clip(lower=0)
        
        adjusted_forecast["TYPE"] = "Adjusted"
        
        return adjusted_forecast

# --- Main Streamlit Application ---
def main():
    st.title("📈 Demand Forecasting with AI Insights")
    st.markdown("### Upload your retail demand data and get ML-powered forecasts and AI insights.")

    safety_pct = 0

    # Sidebar for file upload and parameters
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df["WEEK"] = pd.to_datetime(df["WEEK"])  # Ensure datetime for edit function

            # --- Add the edit function here ---
            if "edited_df" in st.session_state:
                df = st.session_state["edited_df"]
            if "edited_forecast_df" in st.session_state:
                edited_forecast_df = st.session_state["edited_forecast_df"]
            else:
                edited_forecast_df = None
            
            st.sidebar.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            forecaster = DemandForecaster()
            df = forecaster.preprocess_data(df)
            if df is None:
                st.stop()
            
            # st.sidebar.header("🎯 Forecast Configuration")
            # model_type = st.sidebar.selectbox("Forecast Model", ["Random Forest", "Linear Regression"])
            model_type = "Random Forest"  # Default to Random Forest for POC
            # st.sidebar.markdown("Select the model type for forecasting. Random Forest is generally more robust, while Linear Regression is simpler and faster.")

            # Target Variable is set only to Units, no need for any input.
            target_col = "UNITS"
            
            forecast_periods = st.sidebar.slider("Forecast Periods (weeks)", 4, 52, 4)
                                
            st.sidebar.header("🚛 Supply Chain Parameters")
            lead_time_weeks = st.sidebar.slider("Lead Time (weeks)", 0, 12, 0, 
                                              help="Time between order placement and delivery")
            
            # Ollama Model Selection
            ollama_model = "llama3"
            
            st.header("🔮 Demand Forecasting Results")
            
            with st.spinner("Generating forecasts..."):
                # --- Forecast by each product individually ---
                if "PRODUCT" in df.columns:
                    product_forecasts = []
                    product_historicals = []
                    for product in sorted(df["PRODUCT"].astype(str).unique()):
                        product_df = df[df["PRODUCT"].astype(str) == str(product)].copy()
                        forecast_data = forecaster.create_forecast_features(product_df, target_col)
                        prod_forecast, prod_historical = forecaster.train_forecast_model(
                            forecast_data, target_col, forecast_periods, model_type
                        )
                        if prod_forecast is not None:
                            prod_forecast["PRODUCT"] = product
                            product_forecasts.append(prod_forecast)
                        if prod_historical is not None:
                            prod_historical["PRODUCT"] = product
                            product_historicals.append(prod_historical)
                    if product_forecasts:
                        original_forecast = pd.concat(product_forecasts, ignore_index=True)
                    else:
                        original_forecast = None
                    if product_historicals:
                        historical_data = pd.concat(product_historicals, ignore_index=True)
                    else:
                        historical_data = None
                else:
                    forecast_data = forecaster.create_forecast_features(df, target_col)
                    original_forecast, historical_data = forecaster.train_forecast_model(
                        forecast_data, target_col, forecast_periods, model_type
                    )

                if original_forecast is not None:
                    adjusted_forecast = forecaster.apply_supply_chain_adjustments(
                        original_forecast, lead_time_weeks, safety_pct
                    )

                    # --- REMOVE Inventory Forecast Calculation ---
                    # Instead, set INVENTORY_FORECAST to 0 for all records
                    adjusted_forecast = adjusted_forecast.copy()
                    adjusted_forecast["INVENTORY_FORECAST"] = 0

                    st.sidebar.header("📅 Chart Display Settings")
                    forecast_weeks = original_forecast["WEEK"].sort_values().unique()
                    min_week = forecast_weeks[0].to_pydatetime()
                    max_week = forecast_weeks[-1].to_pydatetime()

                    # --- Product Filter for Charts (ALWAYS set before any table or chart) ---
                    product_filter = None
                    if "PRODUCT" in df.columns:
                        all_products = sorted(df["PRODUCT"].astype(str).unique())
                        product_filter = st.sidebar.multiselect(
                            "Filter Products for Chart Display",
                            all_products,
                            default=all_products,
                            key="chart_product_filter"
                        )
                    else:
                        product_filter = None

                    # --- Store filter in session state for sidebar/simulator sync ---
                    st.session_state["active_product_filter"] = product_filter

                    display_range = st.sidebar.slider(
                        "Select Forecasted Date Range to Display",
                        min_value=min_week,
                        max_value=max_week,
                        value=(min_week, max_week),
                        step=pd.Timedelta(weeks=1),
                        format="YYYY-MM-DD"
                    )

                    # --- FILTER ALL DATAFRAMES BY DATE RANGE ---
                    def filter_by_range(df, week_col, start_date, end_date):
                        if df is None or len(df) == 0:
                            return df
                        df = df.copy()
                        # Fix: Accept both 'Week' and 'WEEK' as valid columns, fallback to first found
                        possible_week_cols = [week_col, "Week", "WEEK"]
                        actual_week_col = None
                        for col in possible_week_cols:
                            if col in df.columns:
                                actual_week_col = col
                                break
                        if actual_week_col is None:
                            raise KeyError(f"No valid week column found in DataFrame. Tried: {possible_week_cols}")
                        df[actual_week_col] = pd.to_datetime(df[actual_week_col])
                        return df[(df[actual_week_col] >= pd.to_datetime(start_date)) & (df[actual_week_col] <= pd.to_datetime(end_date))]
                    
                    start_week, end_week = display_range[0], display_range[1]

                    # Apply to all relevant DataFrames
                    original_forecast = filter_by_range(original_forecast, "WEEK", start_week, end_week)
                    adjusted_forecast = filter_by_range(adjusted_forecast, "WEEK", start_week, end_week)
                    historical_data = filter_by_range(historical_data, "WEEK", start_week, end_week) if historical_data is not None else None

                    # For *_display DataFrames, always assign from filtered base if not defined
                    try:
                        original_forecast_display
                    except NameError:
                        original_forecast_display = original_forecast.copy()
                    original_forecast_display = filter_by_range(original_forecast_display, "WEEK", start_week, end_week)

                    try:
                        adjusted_forecast_display
                    except NameError:
                        adjusted_forecast_display = adjusted_forecast.copy()
                    adjusted_forecast_display = filter_by_range(adjusted_forecast_display, "WEEK", start_week, end_week)

                    try:
                        display_adjusted
                    except NameError:
                        display_adjusted = adjusted_forecast.copy()
                    display_adjusted = filter_by_range(display_adjusted, "Week", start_week, end_week)

                    if "sim_base_df" in st.session_state and st.session_state["sim_base_df"] is not None:
                        st.session_state["sim_base_df"] = filter_by_range(st.session_state["sim_base_df"], "Week", start_week, end_week)

                    # Only filter *_display if they already exist, else assign from base
                    if 'original_forecast_display' in locals():
                        if 'PRODUCT' in original_forecast.columns:
                            original_forecast_display = filter_by_range(original_forecast_display, "WEEK", start_week, end_week)
                        else:
                            original_forecast_display = filter_by_range(original_forecast, "WEEK", start_week, end_week)
                    else:
                        original_forecast_display = filter_by_range(original_forecast, "WEEK", start_week, end_week)

                    if 'adjusted_forecast_display' in locals():
                        if 'PRODUCT' in adjusted_forecast.columns:
                            adjusted_forecast_display = filter_by_range(adjusted_forecast_display, "WEEK", start_week, end_week)
                        else:
                            adjusted_forecast_display = filter_by_range(adjusted_forecast, "WEEK", start_week, end_week)
                    else:
                        adjusted_forecast_display = filter_by_range(adjusted_forecast, "WEEK", start_week, end_week)

                    if historical_data is not None and 'WEEK' in historical_data.columns:
                        historical_data = filter_by_range(historical_data, "WEEK", start_week, end_week)

                    # For display_adjusted and simulator tables, use "Week" as the column name
                    if 'display_adjusted' in locals():
                        display_adjusted = filter_by_range(display_adjusted, "Week", start_week, end_week)

                    # --- Call edit_forecast_data here, after adjusted_forecast is defined and display_range is set ---
                    edit_forecast_data(adjusted_forecast, (start_week, end_week), df=df)

                    # --- Always filter all tables and charts using the same product_filter ---
                    def filter_all(df, product_col="PRODUCT"):
                        pf = st.session_state.get("active_product_filter", None)
                        if pf is not None and product_col in df.columns:
                            return df[df[product_col].astype(str).isin(pf)]
                        return df

                    original_forecast = filter_all(original_forecast, "PRODUCT")
                    adjusted_forecast = filter_all(adjusted_forecast, "PRODUCT")
                    if 'PRODUCT' in original_forecast.columns:
                        original_forecast_display = filter_all(original_forecast, "PRODUCT")
                    else:
                        original_forecast_display = original_forecast.copy()
                    if 'PRODUCT' in adjusted_forecast.columns:
                        adjusted_forecast_display = filter_all(adjusted_forecast, "PRODUCT")
                    else:
                        adjusted_forecast_display = adjusted_forecast.copy()
                    if historical_data is not None and 'PRODUCT' in historical_data.columns:
                        historical_data = filter_all(historical_data, "PRODUCT")
                    else:
                        historical_data = historical_data
                    # For chart aggregation
                    def filter_display(df):
                        if 'PRODUCT' in df.columns:
                            return filter_all(df, "PRODUCT")
                        return df
                    # For all display tables
                    # (original_display, adjusted_display) are created below

                    # --- Prepare display plots and y-axis range before both charts ---
                    # --- Use Simulator values for chart if available ---
                    # --- Simulator Table logic ---
                    display_adjusted = adjusted_forecast.copy()
                    display_adjusted["Lead Time (weeks)"] = lead_time_weeks
                    if "PRODUCT" not in display_adjusted.columns and "PRODUCT" in df.columns:
                        display_adjusted["PRODUCT"] = df["PRODUCT"].iloc[0]
                    display_adjusted["WEEK"] = pd.to_datetime(display_adjusted["WEEK"]).dt.strftime("%Y-%m-%d")
                    display_adjusted = display_adjusted.round(0)

                    # --- Ensure all combinations of Product and Week are present ---
                    if "PRODUCT" in df.columns:
                        all_products = sorted(df["PRODUCT"].astype(str).unique())
                        all_weeks = sorted(adjusted_forecast["WEEK"].dt.strftime("%Y-%m-%d").unique())
                        # Apply product filter to all_products
                        if product_filter is not None:
                            all_products = [p for p in all_products if p in product_filter]
                        idx = pd.MultiIndex.from_product([all_weeks, all_products], names=["WEEK", "PRODUCT"])
                        display_adjusted["PRODUCT"] = display_adjusted["PRODUCT"].astype(str)
                        display_adjusted = display_adjusted.set_index(["WEEK", "PRODUCT"])
                        display_adjusted = display_adjusted.reindex(idx).reset_index()
                        display_adjusted["ADJUSTED_FORECAST"] = display_adjusted["ADJUSTED_FORECAST"].fillna(0)
                        display_adjusted["INVENTORY_FORECAST"] = 0
                        display_adjusted["Lead Time (weeks)"] = display_adjusted["Lead Time (weeks)"].fillna(lead_time_weeks)
                    else:
                        all_weeks = sorted(adjusted_forecast["WEEK"].dt.strftime("%Y-%m-%d").unique())
                        display_adjusted = display_adjusted.set_index("WEEK").reindex(all_weeks).reset_index()
                        display_adjusted["ADJUSTED_FORECAST"] = display_adjusted["ADJUSTED_FORECAST"].fillna(0)
                        display_adjusted["INVENTORY_FORECAST"] = 0
                        display_adjusted["Lead Time (weeks)"] = display_adjusted["Lead Time (weeks)"].fillna(lead_time_weeks)
                        if "PRODUCT" not in display_adjusted.columns:
                            display_adjusted["PRODUCT"] = ""

                    # --- Filter table by product filter ---
                    display_adjusted = filter_all(display_adjusted, "PRODUCT")
                    display_adjusted = display_adjusted[["WEEK", "PRODUCT", "ADJUSTED_FORECAST", "INVENTORY_FORECAST", "Lead Time (weeks)"]]
                    display_adjusted.columns = ["Week", "Product", "Base Demand Forecast", "Inventory", "Lead Time (weeks)"]

                    # --- Set the first week Inventory value to 5 for each product ---
                    if "Product" in display_adjusted.columns and "Week" in display_adjusted.columns:
                        for product in display_adjusted["Product"].unique():
                            product_mask = display_adjusted["Product"] == product
                            product_weeks = display_adjusted.loc[product_mask, "Week"]
                            if not product_weeks.empty:
                                first_week = product_weeks.min()
                                idx = display_adjusted.index[(display_adjusted["Product"] == product) & (display_adjusted["Week"] == first_week)]
                                if not idx.empty:
                                    display_adjusted.loc[idx, "Inventory"] = 5

                    display_adjusted.index = display_adjusted.index + 1

                    # Ensure correct types
                    display_adjusted["Week"] = pd.to_datetime(display_adjusted["Week"])
                    display_adjusted["Product"] = display_adjusted["Product"].astype(str)
                    display_adjusted = display_adjusted.sort_values(["Product", "Week"]).reset_index(drop=True)

                    # Set first week's inventory to 5 for each product
                    for product in display_adjusted["Product"].unique():
                        product_mask = display_adjusted["Product"] == product
                        product_weeks = display_adjusted.loc[product_mask, "Week"]
                        if not product_weeks.empty:
                            first_week = product_weeks.min()
                            idx_first = display_adjusted.index[(display_adjusted["Product"] == product) & (display_adjusted["Week"] == first_week)]
                            display_adjusted.loc[idx_first, "Inventory"] = 5

                    # Calculate inventory for subsequent weeks
                    for product in display_adjusted["Product"].unique():
                        product_mask = display_adjusted["Product"] == product
                        product_df = display_adjusted[product_mask].sort_values("Week")
                        for i in range(1, len(product_df)):
                            prev_idx = product_df.index[i - 1]
                            curr_idx = product_df.index[i]
                            prev_inventory = display_adjusted.loc[prev_idx, "Inventory"]
                            curr_demand = display_adjusted.loc[curr_idx, "Base Demand Forecast"]
                            display_adjusted.loc[curr_idx, "Inventory"] = prev_inventory - curr_demand

                    display_adjusted = filter_all(display_adjusted, "Product")

                    # --- CHARTS: Use Simulator Table values for Adjusted Forecast if edits exist ---
                    # Use sim_base for adjusted_display if edits exist, else use adjusted_forecast_display
                    sim_base_for_chart = st.session_state.get("sim_base_df", None)
                    if sim_base_for_chart is not None and not sim_base_for_chart.empty:
                        adjusted_display_for_chart = sim_base_for_chart.copy()
                        adjusted_display_for_chart["WEEK"] = adjusted_display_for_chart["Week"]
                        adjusted_display_for_chart["PRODUCT"] = adjusted_display_for_chart["Product"]
                        adjusted_display_for_chart["ADJUSTED_FORECAST"] = adjusted_display_for_chart["New Demand Forecast"]
                    else:
                        adjusted_display_for_chart = adjusted_forecast_display.copy()
                        if "WEEK" not in adjusted_display_for_chart.columns and "Week" in adjusted_display_for_chart.columns:
                            adjusted_display_for_chart["WEEK"] = adjusted_display_for_chart["Week"]
                        if "PRODUCT" not in adjusted_display_for_chart.columns and "Product" in adjusted_display_for_chart.columns:
                            adjusted_display_for_chart["PRODUCT"] = adjusted_display_for_chart["Product"]
                        adjusted_display_for_chart["ADJUSTED_FORECAST"] = adjusted_display_for_chart["ADJUSTED_FORECAST"]
                    
                    # --- Save original simulator table for reset functionality (only once per session or when data changes) ---
                    if "sim_base_original" not in st.session_state or st.session_state.get("sim_base_original_hash", None) != hash(display_adjusted.to_csv(index=False)):
                        save_simulator_original(display_adjusted)
                        st.session_state["sim_base_original_hash"] = hash(display_adjusted.to_csv(index=False))

                    # --- Add Reset Button in Sidebar ---
                    st.sidebar.markdown("---")
                    if st.sidebar.button("Reset Simulator Table"):
                        reset_simulator_to_original()
                    
                    # --- Prepare chart dataframes ---
                    # Always filter by product
                    adjusted_display_for_chart = filter_all(adjusted_display_for_chart, "PRODUCT")
                    original_forecast_display = filter_all(original_forecast_display, "PRODUCT")

                    # --- Chart grouping logic ---
                    if forecast_periods <= 12:
                        original_display = original_forecast_display.copy()
                        adjusted_display = adjusted_display_for_chart.copy()
                        x_col = "WEEK"
                        x_title = "Week"
                    elif forecast_periods < 52:
                        original_display = original_forecast_display.copy()
                        adjusted_display = adjusted_display_for_chart.copy()
                        original_display["MONTH"] = pd.to_datetime(original_display["WEEK"]).dt.to_period("M").dt.to_timestamp()
                        adjusted_display["MONTH"] = pd.to_datetime(adjusted_display["WEEK"]).dt.to_period("M").dt.to_timestamp()
                        x_col = "MONTH"
                        x_title = "Month"
                        if "PRODUCT" in original_display.columns:
                            orig_agg = {col: "sum" for col in ["FORECAST"] if col in original_display.columns}
                            adj_agg = {col: "sum" for col in ["ADJUSTED_FORECAST"] if col in adjusted_display.columns}
                            original_display = original_display.groupby([x_col, "PRODUCT"], as_index=False).agg(orig_agg)
                            adjusted_display = adjusted_display.groupby([x_col, "PRODUCT"], as_index=False).agg(adj_agg)
                        else:
                            if "FORECAST" in original_display.columns:
                                original_display = original_display.groupby(x_col, as_index=False)["FORECAST"].sum()
                            if "ADJUSTED_FORECAST" in adjusted_display.columns:
                                adjusted_display = adjusted_display.groupby(x_col, as_index=False)["ADJUSTED_FORECAST"].sum()
                    elif forecast_periods == 52:
                        original_display = original_forecast_display.copy()
                        adjusted_display = adjusted_display_for_chart.copy()
                        original_display["QUARTER"] = pd.to_datetime(original_display["WEEK"]).dt.to_period("Q").dt.to_timestamp()
                        adjusted_display["QUARTER"] = pd.to_datetime(adjusted_display["WEEK"]).dt.to_period("Q").dt.to_timestamp()
                        x_col = "QUARTER"
                        x_title = "Quarter"
                        if "PRODUCT" in original_display.columns:
                            orig_agg = {col: "sum" for col in ["FORECAST"] if col in original_display.columns}
                            adj_agg = {col: "sum" for col in ["ADJUSTED_FORECAST"] if col in adjusted_display.columns}
                            original_display = original_display.groupby([x_col, "PRODUCT"], as_index=False).agg(orig_agg)
                            adjusted_display = adjusted_display.groupby([x_col, "PRODUCT"], as_index=False).agg(adj_agg)
                        else:
                            if "FORECAST" in original_display.columns:
                                original_display = original_display.groupby(x_col, as_index=False)["FORECAST"].sum()
                            if "ADJUSTED_FORECAST" in adjusted_display.columns:
                                adjusted_display = adjusted_display.groupby(x_col, as_index=False)["ADJUSTED_FORECAST"].sum()
                    else:
                        original_display = original_forecast_display.copy()
                        adjusted_display = adjusted_display_for_chart.copy()
                        x_col = "WEEK"
                        x_title = "Week"

                    # --- Prepare display plots and y-axis range before both charts ---
                    original_display_plot = original_display.copy()
                    adjusted_display_plot = adjusted_display.copy()

                    # Ensure columns exist for plotting
                    if "FORECAST" in original_display_plot.columns:
                        original_display_plot["FORECAST"] = np.ceil(original_display_plot["FORECAST"]).astype(int)
                    if "ADJUSTED_FORECAST" in adjusted_display_plot.columns:
                        adjusted_display_plot["ADJUSTED_FORECAST"] = np.ceil(adjusted_display_plot["ADJUSTED_FORECAST"]).astype(int)

                    # Collect y-values from both historical and forecast for shared axis
                    y_values = []
                    if historical_data is not None:
                        y_values.extend(historical_data["ACTUAL"].values)
                    if "FORECAST" in original_display_plot.columns:
                        y_values.extend(original_display_plot["FORECAST"].values)
                    if "ADJUSTED_FORECAST" in adjusted_display_plot.columns:
                        y_values.extend(adjusted_display_plot["ADJUSTED_FORECAST"].values)
                    y_values = [v for v in y_values if pd.notnull(v)]
                    if y_values:
                        y_min = int(np.floor(min(y_values)))
                        y_max = int(np.ceil(max(y_values)))
                        if y_min == y_max:
                            y_max = y_min + 1  # Ensure some range
                    else:
                        y_min, y_max = 0, 1

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📈 Historical Data")
                        hist_chart_type = st.sidebar.radio(
                            "Historical Chart Type", 
                            ["Line Chart", "Bar Chart"], 
                            index=1,
                            key="hist_chart_type",
                            horizontal=True
                        )

                        # --- FIX: Use the original CSV data for historical chart, not the processed/filtered model data ---
                        filtered_df = df.copy()

                        # Ensure WEEK is datetime for filtering (robust against already-converted columns)
                        filtered_df["WEEK"] = pd.to_datetime(filtered_df["WEEK"], errors="coerce")

                        # Remove rows with missing WEEK or UNITS
                        filtered_df = filtered_df[pd.notnull(filtered_df["WEEK"]) & pd.notnull(filtered_df["UNITS"])]

                        # Filter by product if applicable
                        if product_filter is not None and "PRODUCT" in filtered_df.columns:
                            filtered_df = filtered_df[filtered_df["PRODUCT"].astype(str).isin(product_filter)]

                        # --- FIX: If UNITS column is string/object, convert to numeric ---
                        if filtered_df["UNITS"].dtype not in [np.float64, np.int64]:
                            filtered_df["UNITS"] = pd.to_numeric(filtered_df["UNITS"], errors="coerce")
                        filtered_df = filtered_df[pd.notnull(filtered_df["UNITS"])]

                        historical_fig = go.Figure()
                        if not filtered_df.empty:
                            if "PRODUCT" in filtered_df.columns:
                                for product in sorted(filtered_df["PRODUCT"].astype(str).unique()):
                                    prod_hist = filtered_df[filtered_df["PRODUCT"].astype(str) == str(product)]
                                    if not prod_hist.empty:
                                        if hist_chart_type == "Line Chart":
                                            historical_fig.add_trace(go.Scatter(
                                                x=prod_hist["WEEK"],
                                                y=prod_hist["UNITS"],
                                                mode="lines+markers",
                                                name=f"Historical Actual - {product}",
                                                line=dict(width=2),
                                                marker=dict(size=6),
                                                opacity=0.8
                                            ))
                                        elif hist_chart_type == "Bar Chart":
                                            historical_fig.add_trace(go.Bar(
                                                x=prod_hist["WEEK"],
                                                y=prod_hist["UNITS"],
                                                name=f"Historical Actual - {product}",
                                                opacity=0.8
                                            ))
                            else:
                                if hist_chart_type == "Line Chart":
                                    historical_fig.add_trace(go.Scatter(
                                        x=filtered_df["WEEK"],
                                        y=filtered_df["UNITS"],
                                        mode="lines+markers",
                                        name="Historical Actual",
                                        line=dict(color="gray", width=2),
                                        marker=dict(size=6),
                                        opacity=0.8
                                    ))
                                elif hist_chart_type == "Bar Chart":
                                    historical_fig.add_trace(go.Bar(
                                        x=filtered_df["WEEK"],
                                        y=filtered_df["UNITS"],
                                        name="Historical Actual",
                                        marker_color="gray",
                                        opacity=0.8
                                    ))
                        else:
                            st.info("No historical data available for the selected date range or filters.")

                        historical_fig.update_layout(
                            title="Historical Actuals - UNITS",
                            xaxis_title="Week",
                            yaxis_title="Units Sold",
                            hovermode="x unified",
                            height=500
                        )
                        # Set y-axis range if there is data
                        if not filtered_df.empty:
                            y_min = int(np.floor(filtered_df["UNITS"].min()))
                            y_max = int(np.ceil(filtered_df["UNITS"].max()))
                            if y_min == y_max:
                                y_max = y_min + 1
                            historical_fig.update_yaxes(tickformat="d", range=[y_min, y_max])

                        if len(historical_fig.data) > 0:
                            st.plotly_chart(historical_fig, use_container_width=True)
                        else:
                            st.info("No data to display in the historical chart for the selected filters.")
                                            
                    with col2:
                        st.subheader("📈 Original vs Adjusted Forecasts")
                        forecast_chart_type = st.sidebar.radio(
                            "Forecast Chart Type", 
                            ["Line Chart", "Bar Chart"], 
                            index=1,
                            key="forecast_chart_type",
                            horizontal=True
                        )

                        forecast_fig = go.Figure()

                        # --- Plot by product if available, else aggregate ---
                        if "PRODUCT" in original_display_plot.columns:
                            for product in sorted(original_display_plot["PRODUCT"].astype(str).unique()):
                                prod_orig = original_display_plot[original_display_plot["PRODUCT"].astype(str) == str(product)]
                                prod_adj = adjusted_display_plot[adjusted_display_plot["PRODUCT"].astype(str) == str(product)]
                                if forecast_chart_type == "Line Chart":
                                    forecast_fig.add_trace(go.Scatter(
                                        x=prod_orig[x_col],
                                        y=prod_orig["FORECAST"] if "FORECAST" in prod_orig.columns else prod_orig["ADJUSTED_FORECAST"],
                                        mode="lines+markers",
                                        name=f"Original Forecast - {product}",
                                        line=dict(width=3),
                                        marker=dict(size=6)
                                    ))
                                    forecast_fig.add_trace(go.Scatter(
                                        x=prod_adj[x_col],
                                        y=prod_adj["ADJUSTED_FORECAST"] if "ADJUSTED_FORECAST" in prod_adj.columns else prod_adj["FORECAST"],
                                        mode="lines+markers",
                                        name=f"Adjusted Forecast - {product}",
                                        line=dict(width=3, dash="dash"),
                                        marker=dict(size=6)
                                    ))
                                elif forecast_chart_type == "Bar Chart":
                                    forecast_fig.add_trace(go.Bar(
                                        x=prod_orig[x_col],
                                        y=prod_orig["FORECAST"] if "FORECAST" in prod_orig.columns else prod_orig["ADJUSTED_FORECAST"],
                                        name=f"Original Forecast - {product}",
                                        opacity=0.7
                                    ))
                                    forecast_fig.add_trace(go.Bar(
                                        x=prod_adj[x_col],
                                        y=prod_adj["ADJUSTED_FORECAST"] if "ADJUSTED_FORECAST" in prod_adj.columns else prod_adj["FORECAST"],
                                        name=f"Adjusted Forecast - {product}",
                                        opacity=0.7
                                    ))
                        else:
                            if forecast_chart_type == "Line Chart":
                                forecast_fig.add_trace(go.Scatter(
                                    x=original_display_plot[x_col],
                                    y=original_display_plot["FORECAST"] if "FORECAST" in original_display_plot.columns else original_display_plot["ADJUSTED_FORECAST"],
                                    mode="lines+markers",
                                    name="Original Forecast",
                                    line=dict(color="blue", width=3),
                                    marker=dict(size=6)
                                ))
                                forecast_fig.add_trace(go.Scatter(
                                    x=adjusted_display_plot[x_col],
                                    y=adjusted_display_plot["ADJUSTED_FORECAST"] if "ADJUSTED_FORECAST" in adjusted_display_plot.columns else adjusted_display_plot["FORECAST"],
                                    mode="lines+markers",
                                    name="Adjusted Forecast (Lead Time + Buffer)",
                                    line=dict(color="red", width=3, dash="dash"),
                                    marker=dict(size=6)
                                ))
                            elif forecast_chart_type == "Bar Chart":
                                forecast_fig.add_trace(go.Bar(
                                    x=original_display_plot[x_col],
                                    y=original_display_plot["FORECAST"] if "FORECAST" in original_display_plot.columns else original_display_plot["ADJUSTED_FORECAST"],
                                    name="Original Forecast",
                                    marker_color="blue",
                                    opacity=0.7
                                ))
                                forecast_fig.add_trace(go.Bar(
                                    x=adjusted_display_plot[x_col],
                                    y=adjusted_display_plot["ADJUSTED_FORECAST"] if "ADJUSTED_FORECAST" in adjusted_display_plot.columns else adjusted_display_plot["FORECAST"],
                                    name="Adjusted Forecast (Lead Time + Buffer)",
                                    marker_color="red",
                                    opacity=0.7
                                ))

                        forecast_fig.update_layout(
                            title=f"Demand Forecast Comparison - {target_col}",
                            xaxis_title=x_title,
                            yaxis_title=target_col,
                            barmode="group",
                            height=500
                        )
                        forecast_fig.update_yaxes(tickformat="d", range=[y_min, y_max])

                        st.plotly_chart(forecast_fig, use_container_width=True)

                    st.subheader("📊 Impact Analysis")
                        
                    total_original = original_forecast["FORECAST"].sum()
                    total_adjusted = adjusted_forecast["ADJUSTED_FORECAST"].sum()
                    impact_pct = (total_adjusted - total_original) / total_original * 100 if total_original > 0 else 0

                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        
                    with metrics_col1:
                        st.metric("Original Forecast Total", f"{total_original:,.0f}")
                    with metrics_col2:
                        st.metric("Lead Time Adjustment", f"{lead_time_weeks} weeks")
                    with metrics_col3:
                        st.metric("Adjusted Forecast Total", f"{total_adjusted:,.0f}"
                                    )
                    with metrics_col4:
                        st.metric("Total Impact", f"{impact_pct:+.1f}%")

                    #Tables
                    tab1, tab2, tab3 = st.tabs(["Base Forecast", "Simulator", "Comparison"])

                    with tab1:
                        st.subheader("Base Forecast Details")
                        display_adjusted_for_table = display_adjusted.copy()
                        display_adjusted_for_table.index = np.arange(1, len(display_adjusted_for_table) + 1)
                        # Columns: Week, Product, Base Demand Forecast, Inventory, Lead Time (weeks)
                        st.dataframe(display_adjusted_for_table[["Week", "Product", "Base Demand Forecast", "Inventory", "Lead Time (weeks)"]], use_container_width=True)

                    with tab2:
                        st.subheader("Simulator")
                        edit_changes = st.session_state.get("edit_changes", [])

                        # Start from the authoritative Base Forecast table
                        sim_base = display_adjusted.copy()

                        # --- ALWAYS ensure columns exist before assignment, even after reset ---
                        if "New Demand Forecast" not in sim_base.columns:
                            sim_base["New Demand Forecast"] = sim_base["Base Demand Forecast"]
                        if "New Inventory Forecast" not in sim_base.columns:
                            sim_base["New Inventory Forecast"] = sim_base["Inventory"]

                        # --- ROUND before applying edits and recalculation ---
                        sim_base["New Demand Forecast"] = np.round(sim_base["New Demand Forecast"])
                        sim_base["New Inventory Forecast"] = np.round(sim_base["New Inventory Forecast"])

                        # Apply user edits from the sidebar
                        if edit_changes:
                            for edit in edit_changes:
                                week = pd.to_datetime(edit.get("Demand Forecast Week"))
                                product = str(edit.get("Product", ""))
                                table = edit.get("Table") if "Table" in edit else edit.get("Field Type")
                                new_val = edit.get("New Unit")
                                mask = (sim_base["Week"] == week)
                                if product:
                                    mask &= (sim_base["Product"].astype(str) == product)
                                if table == "Unit":
                                    sim_base.loc[mask, "New Demand Forecast"] = new_val
                                elif table == "Inventory":
                                    sim_base.loc[mask, "New Inventory Forecast"] = new_val

                            # --- ROUND again after edits, before recalculation ---
                            sim_base["New Demand Forecast"] = np.round(sim_base["New Demand Forecast"])
                            sim_base["New Inventory Forecast"] = np.round(sim_base["New Inventory Forecast"])

                            # Recalculate "New Inventory Forecast" for all products after edits
                            sim_base = recalculate_inventory(
                                sim_base,
                                demand_col="New Demand Forecast",
                                inventory_col="New Inventory Forecast"
                            )

                            # --- ROUND after recalculation ---
                            sim_base["New Demand Forecast"] = np.round(sim_base["New Demand Forecast"])
                            sim_base["New Inventory Forecast"] = np.round(sim_base["New Inventory Forecast"])

                        # --- Filter simulator table by product filter ---
                        sim_base = filter_all(sim_base, "Product")

                        sim_base.index = sim_base.index + 1
                        sim_base = filter_all(sim_base, "Product")

                        # --- Store the updated Simulator table for sidebar access ---
                        st.session_state["sim_base_df"] = sim_base.copy()

                        st.dataframe(sim_base[["Week", "Product", "Base Demand Forecast", "Inventory", "Lead Time (weeks)", "New Demand Forecast", "New Inventory Forecast"]], use_container_width=True)

                        # Show edit history below the simulator table
                        if edit_changes:
                            edit_history_df = pd.DataFrame(edit_changes)
                            # Remove 'Field', 'Type', and 'Field Type' columns if present
                            edit_history_df = edit_history_df.drop(
                                columns=[col for col in ["Field", "Type", "Field Type"] if col in edit_history_df.columns],
                                errors="ignore"
                            )
                            # Rename columns as requested
                            edit_history_df = edit_history_df.rename(
                                columns={
                                    "Original Unit": "Original Value",
                                    "New Unit": "New Value"
                                }
                            )
                            # Set column order
                            columns_order = ["Value Type", "Demand Forecast Week", "Product", "Original Value", "New Value"]
                            edit_history_df = edit_history_df[[col for col in columns_order if col in edit_history_df.columns]]
                            edit_history_df.index = edit_history_df.index + 1
                            st.markdown("**Edit History:**")
                            st.dataframe(edit_history_df, use_container_width=True)
                        else:
                            st.info("No edits have been made to forecast data yet.")

                    # --- Comparison Table ---
                    with tab3:
                        st.subheader("Side-by-Side Comparison")
                        sim_base = st.session_state.get("sim_base_df", None)
                        if sim_base is not None and not sim_base.empty:
                            # Prepare base DataFrame for comparison (original/base values)
                            base_df = display_adjusted.copy()
                            base_df = base_df[["Week", "Product", "Lead Time (weeks)", "Base Demand Forecast", "Inventory"]]
                            # Merge with sim_base to get new values
                            sim_base_comp = sim_base[["Week", "Product", "New Demand Forecast", "New Inventory Forecast"]].copy()
                            comparison_df = pd.merge(
                                base_df,
                                sim_base_comp,
                                on=["Week", "Product"],
                                how="left",
                                suffixes=("", "_new")
                            )
                            # Calculate differences
                            comparison_df["Difference (Demand)"] = comparison_df["Base Demand Forecast"] - comparison_df["New Demand Forecast"]
                            comparison_df["Difference (Inventory)"] = comparison_df["New Inventory Forecast"] - comparison_df["Inventory"]

                            # Arrange columns as specified
                            comparison_df = comparison_df[
                                [
                                    "Week",
                                    "Product",
                                    "Lead Time (weeks)",
                                    "Base Demand Forecast",
                                    "New Demand Forecast",
                                    "Difference (Demand)",
                                    "Inventory",
                                    "New Inventory Forecast",
                                    "Difference (Inventory)"
                                ]
                            ]
                            # Rename columns for clarity
                            comparison_df = comparison_df.rename(columns={
                                "Lead Time (weeks)": "Lead Time (weeks)",
                                "Base Demand Forecast": "Base Forecast",
                                "New Demand Forecast": "New Demand Forecast",
                                "Inventory": "Inventory",
                                "New Inventory Forecast": "New Inventory",
                                "Difference (Demand)": "Difference (Demand)",
                                "Difference (Inventory)": "Difference (Inventory)"
                            })
                            # Round numeric columns for display
                            for col in ["Base Forecast", "New Demand Forecast", "Difference (Demand)", "Inventory", "New Inventory", "Difference (Inventory)"]:
                                if col in comparison_df.columns:
                                    comparison_df[col] = np.round(comparison_df[col], 2)
                        else:
                            # Fallback: show only base columns if sim_base is not available
                            comparison_df = display_adjusted.copy()
                            comparison_df["New Demand Forecast"] = np.nan
                            comparison_df["Difference (Demand)"] = np.nan
                            comparison_df["New Inventory"] = np.nan
                            comparison_df["Difference (Inventory)"] = np.nan
                            comparison_df = comparison_df[
                                [
                                    "Week",
                                    "Product",
                                    "Lead Time (weeks)",
                                    "Base Demand Forecast",
                                    "New Demand Forecast",
                                    "Difference (Demand)",
                                    "Inventory",
                                    "New Inventory",
                                    "Difference (Inventory)"
                                ]
                            ]
                            comparison_df = comparison_df.rename(columns={
                                "Lead Time (weeks)": "Lead Time (weeks)",
                                "Base Demand Forecast": "Base Forecast",
                                "Inventory": "Inventory"
                            })

                        # --- Filter comparison table by product filter ---
                        comparison_df = filter_all(comparison_df, "Product")
                        comparison_df.index = comparison_df.index + 1
                        comparison_df = filter_all(comparison_df, "Product")
                        st.dataframe(comparison_df, use_container_width=True)
                        
                    # --- AI Insights Integration ---
                    st.markdown("---")
                    st.subheader("🤖 AI-Powered Insights")

                    # Prepare data for Ollama prompt
                    # Use adjusted_forecast for insights
                    ollama_forecast_data = adjusted_forecast[["WEEK", "ADJUSTED_FORECAST", "PRODUCT"]].copy() if "PRODUCT" in adjusted_forecast.columns else adjusted_forecast[["WEEK", "ADJUSTED_FORECAST"]].copy()
                    if "PRODUCT" in ollama_forecast_data.columns:
                        ollama_forecast_data.columns = ["WEEK", "UNITS", "PRODUCT"]
                    else:
                        ollama_forecast_data.columns = ["WEEK", "UNITS"]

                    # Determine filtered products for chart display
                    filtered_products = product_filter if product_filter is not None else []
                    multi_product_mode = len(filtered_products) > 1

                    # --- Generate prompt for Ollama using the correct function based on product filter ---
                    if multi_product_mode:
                        # Multi-product: use generate_multi_product_prompt
                        ollama_prompt = generate_multi_product_prompt(
                            forecast_df=display_adjusted[display_adjusted["Product"].isin(filtered_products)],
                            actuals_df=None  # Add actuals_df if available
                        )
                        ollama_context_products = filtered_products
                    else:
                        # Single product or all: use generate_prompt_ollama
                        if "PRODUCT" in df.columns and len(filtered_products) == 1:
                            product_id_for_ollama = filtered_products[0]
                        elif "PRODUCT" in df.columns:
                            product_id_for_ollama = df["PRODUCT"].iloc[0]
                        else:
                            product_id_for_ollama = "Overall Demand"
                        ollama_prompt = generate_prompt_ollama(
                            ollama_forecast_data["WEEK"],
                            ollama_forecast_data["UNITS"],
                            product_id_for_ollama,
                            ollama_forecast_data["WEEK"].tolist()
                        )
                        ollama_context_products = [product_id_for_ollama]

                    if st.button("Generate AI Insights with Ollama"):
                        with st.spinner(f"Generating AI insights using {ollama_model}..."):
                            try:
                                ollama_output = run_ollama_forecast(ollama_prompt, model=ollama_model)
                                historical_units_for_ollama = historical_data["ACTUAL"].values if historical_data is not None else None

                                ollama_forecast_parsed, ollama_markdown_table, ollama_insights = parse_forecast_ollama(
                                    ollama_output, historical_units=historical_units_for_ollama
                                )
                                st.session_state["ollama_insights"] = {
                                    "markdown_table": ollama_markdown_table,
                                    "insights": ollama_insights,
                                    "forecast": ollama_forecast_parsed,
                                    "future_weeks": ollama_forecast_data["WEEK"].tolist(),
                                    "product_id": ollama_context_products,
                                    "multi_product_mode": multi_product_mode,
                                    "ollama_prompt": ollama_prompt
                                }
                            except Exception as e:
                                st.error(f"Error generating Ollama insights: {e}")
                                st.info("Please ensure Ollama is running and the selected model is available.")

                    if "ollama_insights" in st.session_state:
                        ollama_results = st.session_state["ollama_insights"]
                        st.markdown("**Forecast Table from AI:**")
                        st.markdown(ollama_results["markdown_table"])
                        st.markdown("**AI Insights:**")
                        st.markdown(ollama_results["insights"])

                        # --- Chat Section ---
                        st.markdown("---")
                        st.subheader("💬 Chat with AI about Forecasts")
                        if "chat_history_unified" not in st.session_state:
                            st.session_state.chat_history_unified = []

                        for msg in st.session_state.chat_history_unified:
                            st.chat_message(msg["role"]).write(msg["content"])

                        user_input_chat = st.chat_input("Ask the AI about the forecasts or insights...")

                        # --- Prepare edit history context ---
                        edit_changes = st.session_state.get("edit_changes", [])
                        edit_history_text = ""
                        if edit_changes:
                            edit_history_text = (
                                "⚠️ The following edits have been made to the historical or forecasted data by the user:\n"
                                + "\n".join(
                                    [
                                        f"- {e.get('Value Type','')} for {e.get('Product','')} on {e.get('Demand Forecast Week','')} set to {e.get('New Value','')}"
                                        for e in edit_changes
                                    ]
                                )
                                + "\n"
                                "Please consider these changes in your analysis and responses.\n"
                            )

                        # Use the latest edited data for context in chat and AI insights
                        # ollama_forecast_data is already based on adjusted_forecast, which is recalculated after edits

                        if user_input_chat:
                            st.session_state.chat_history_unified.append({"role": "user", "content": user_input_chat})
                            st.chat_message("user").write(user_input_chat)

                            with st.spinner(f"AI is thinking using {ollama_model}..."):
                                chat_prompt = (
                                    edit_history_text
                                    + "User question: "
                                    + user_input_chat
                                )
                                # Use multi-product prompt context if in multi-product mode
                                if ollama_results.get("multi_product_mode", False):
                                    # Use the same multi-product prompt as context for the chat
                                    chat_prompt = ollama_results["ollama_prompt"] + "\n\n" + chat_prompt
                                else:
                                    # Use the single-product prompt context
                                    chat_prompt = build_chat_prompt_ollama(
                                        chat_prompt,
                                        ollama_forecast_data,
                                        ollama_results["future_weeks"],
                                        ollama_results["forecast"]
                                    )
                                response = cached_llm_response(chat_prompt, model=ollama_model)
                            st.session_state.chat_history_unified.append({"role": "assistant", "content": response})
                            st.chat_message("assistant").write(response)

                    st.markdown("---")
                    st.subheader("📝 Feature Descriptions")
                    feature_descriptions = {
                        "WEEK": "The week of the observation (date or week number).",
                        "PRODUCT": "Product identifier or name.",
                        "CATEGORY": "Product category.",
                        "MANUFACTURER": "Manufacturer of the product.",
                        "BRAND": "Brand of the product.",
                        "SUBCATEGORY": "Sub-category of the product.",
                        "SIZE": "Size of the product (if applicable).",
                        "UNITS": "Units sold during the week.",
                        "INVENTORY": "Inventory level at the start/end of the week.",
                    }
                    feature_table = pd.DataFrame({
                        "Feature": df.columns,
                        "Description": [feature_descriptions.get(col, "No description available.") for col in df.columns]
                    })

                    feature_table = feature_table[feature_table["Description"] != "No description available."].reset_index(drop=True)
                    st.dataframe(feature_table, use_container_width=True)

                    required_columns = ["WEEK", "UNITS", "SALES"]
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {missing_columns}")
                        st.stop()

                else:
                    st.error("Unable to generate forecasts. Please check your data quality and try again.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file contains the required columns: WEEK, UNITS, SALES")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        st.subheader("📋 Expected Data Format")
        
        sample_data = pd.DataFrame({
            "WEEK": ["2024-01-01", "2024-01-08", "2024-01-15"],
            "PRODUCT": ["Product_X", "Product_X", "Product_X"],
            "CATEGORY": ["Category_1", "Category_1", "Category_1"],
            "UNITS": [100, 150, 120],
            "INVENTORY": [500, 400, 450]
        })
        
        st.dataframe(sample_data)
        
        st.markdown("""
        ### Key Features:
        - **ML Forecasting**: Linear Regression and Random Forest for time series.
        - **AI Insights**: Powered by Ollama for textual analysis and chat.
        - **Supply Chain Optimization**: Adjust for lead time and safety stock.
        - **Interactive Visualizations**: Real-time updates as you adjust parameters.
        - **Multi-level Filtering**: Analyze by store, product, or category.
        """)

if __name__ == "__main__":
    main()
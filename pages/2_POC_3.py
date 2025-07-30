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

# Set page configuration
st.set_page_config(
    page_title="Demand Forecasting with AI Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    # Sidebar for file upload and parameters
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
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
            
            st.sidebar.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            forecaster = DemandForecaster()
            df = forecaster.preprocess_data(df)
            if df is None:
                st.stop()
            
            st.sidebar.header("🎯 Forecast Configuration")
            model_type = st.sidebar.selectbox("Forecast Model", ["Random Forest", "Linear Regression"])
            st.sidebar.markdown("Select the model type for forecasting. Random Forest is generally more robust, while Linear Regression is simpler and faster.")

            target_options = ["UNITS", "INVENTORY"]
            target_col = st.sidebar.selectbox("Select Target Variable", target_options)
            
            forecast_periods = st.sidebar.slider("Forecast Periods (weeks)", 4, 52, 4)
            
            st.sidebar.header("📊 Data Filters")

            filter_options = []
            if "PRODUCT" in df.columns:
                filter_options.append("Product")
            if "MANUFACTURER" in df.columns:
                filter_options.append("Manufacturer")
            if "BRAND" in df.columns:
                filter_options.append("Brand")
            if "CATEGORY" in df.columns:
                filter_options.append("Category")
            if "SUBCATEGORY" in df.columns:
                filter_options.append("Sub-Category")
            if "SIZE" in df.columns:
                filter_options.append("Size")

            selected_filters = st.sidebar.multiselect(
                "Select Filters to Apply",
                filter_options,
                default=[]
            )

            if "Product" in selected_filters and "PRODUCT" in df.columns:
                products = [str(int(p)) if isinstance(p, (int, float)) and float(p).is_integer() else str(p) for p in sorted(df["PRODUCT"].unique())]
                selected_products = st.sidebar.multiselect("Select Product(s)", products, default=products)
                if selected_products:
                    df = df[df["PRODUCT"].astype(str).isin(selected_products)]

            if "Manufacturer" in selected_filters and "MANUFACTURER" in df.columns:
                manufacturers = sorted(df["MANUFACTURER"].astype(str).unique())
                selected_manufacturers = st.sidebar.multiselect("Select Manufacturer(s)", manufacturers, default=manufacturers)
                if selected_manufacturers:
                    df = df[df["MANUFACTURER"].astype(str).isin(selected_manufacturers)]

            if "Brand" in selected_filters and "BRAND" in df.columns:
                brands = sorted(df["BRAND"].astype(str).unique())
                selected_brands = st.sidebar.multiselect("Select Brand(s)", brands, default=brands)
                if selected_brands:
                    df = df[df["BRAND"].astype(str).isin(selected_brands)]

            if "Category" in selected_filters and "CATEGORY" in df.columns:
                categories = sorted(df["CATEGORY"].astype(str).unique())
                selected_categories = st.sidebar.multiselect("Select Category(s)", categories, default=categories)
                if selected_categories:
                    df = df[df["CATEGORY"].astype(str).isin(selected_categories)]

            if "Sub-Category" in selected_filters and "SUBCATEGORY" in df.columns:
                sub_categories = sorted(df["SUBCATEGORY"].astype(str).unique())
                selected_sub_categories = st.sidebar.multiselect("Select Sub-Category(s)", sub_categories, default=sub_categories)
                if selected_sub_categories:
                    df = df[df["SUBCATEGORY"].astype(str).isin(selected_sub_categories)]

            if "Size" in selected_filters and "SIZE" in df.columns:
                sizes = sorted(df["SIZE"].astype(str).unique())
                selected_sizes = st.sidebar.multiselect("Select Size(s)", sizes, default=sizes)
                if selected_sizes:
                    df = df[df["SIZE"].astype(str).isin(selected_sizes)]
                                
            st.sidebar.header("🚛 Supply Chain Parameters")
            lead_time_weeks = st.sidebar.slider("Lead Time (weeks)", 0, 12, 0, 
                                              help="Time between order placement and delivery")

            safety_pct = st.sidebar.slider("Safety Stock (%)", 0, 100, 0,
                                         help="Percentage adjustment to forecasted demand",
                                         disabled=True)
            
            # Ollama Model Selection
            st.sidebar.header("🤖 AI Insights Configuration")
            ollama_model = st.sidebar.selectbox("Select Ollama Model", ["llama3", "llama2", "mistral"], index=0)
            
            st.header("🔮 Demand Forecasting Results")
            
            with st.spinner("Generating forecasts..."):
                forecast_data = forecaster.create_forecast_features(df, target_col)
                original_forecast, historical_data = forecaster.train_forecast_model(
                    forecast_data, target_col, forecast_periods, model_type
                )
                
                if original_forecast is not None:
                    adjusted_forecast = forecaster.apply_supply_chain_adjustments(
                        original_forecast, lead_time_weeks, safety_pct
                    )

                    st.sidebar.header("📅 Chart Display Settings")
                    forecast_weeks = original_forecast["WEEK"].sort_values().unique()
                    min_week = forecast_weeks[0].to_pydatetime()
                    max_week = forecast_weeks[-1].to_pydatetime()

                    display_range = st.sidebar.slider(
                        "Select Forecasted Date Range to Display",
                        min_value=min_week,
                        max_value=max_week,
                        value=(min_week, max_week),
                        step=pd.Timedelta(weeks=1),
                        format="YYYY-MM-DD"
                    )

                    inventory_change_table = None

                    if target_col == "INVENTORY" and "INVENTORY" in df.columns and "display_range" in locals():
                        first_forecast_week = pd.to_datetime(display_range[0])
                        orig_inventory = 0
                        if "WEEK" in original_forecast.columns and "FORECAST" in original_forecast.columns:
                            forecast_row = original_forecast[original_forecast["WEEK"] == first_forecast_week]
                            if not forecast_row.empty:
                                orig_inventory = int(forecast_row["FORECAST"].iloc[0])
                            else:
                                if not df[df["WEEK"] == first_forecast_week].empty:
                                    orig_inventory = int(df.loc[df["WEEK"] == first_forecast_week, "INVENTORY"].iloc[0])
                                else:
                                    orig_inventory = 0
                        else:
                            if not df[df["WEEK"] == first_forecast_week].empty:
                                orig_inventory = int(df.loc[df["WEEK"] == first_forecast_week, "INVENTORY"].iloc[0])
                            else:
                                orig_inventory = 0

                        changed_inventory = st.sidebar.number_input(
                            f"Inventory for {first_forecast_week.strftime('%Y-%m-%d')}",
                            min_value=0,
                            value=orig_inventory,
                            step=1,
                            key=f"inv_{first_forecast_week}",
                            disabled=True 
                        )

                        if changed_inventory != orig_inventory:
                            df.loc[df["WEEK"] == first_forecast_week, "INVENTORY"] = changed_inventory

                        inventory_change_table = pd.DataFrame({
                            "Week": [first_forecast_week.strftime("%Y-%m-%d")],
                            "Original Inventory": [orig_inventory],
                            "Changed Inventory": [changed_inventory]
                        })

                    original_forecast_display = original_forecast[
                        (original_forecast["WEEK"] >= display_range[0]) & (original_forecast["WEEK"] <= display_range[1])
                    ]
                    adjusted_forecast_display = adjusted_forecast[
                        (adjusted_forecast["WEEK"] >= display_range[0]) & (adjusted_forecast["WEEK"] <= display_range[1])
                    ]

                    if forecast_periods <= 12:
                        original_display = original_forecast_display.copy()
                        adjusted_display = adjusted_forecast_display.copy()
                        x_col = "WEEK"
                        x_title = "Week"
                    elif forecast_periods < 52:
                        original_display = original_forecast_display.copy()
                        adjusted_display = adjusted_forecast_display.copy()
                        original_display["MONTH"] = pd.to_datetime(original_display["WEEK"]).dt.to_period("M").dt.to_timestamp()
                        adjusted_display["MONTH"] = pd.to_datetime(adjusted_display["WEEK"]).dt.to_period("M").dt.to_timestamp()
                        x_col = "MONTH"
                        x_title = "Month"
                        original_display = original_display.groupby(x_col)["FORECAST"].sum().reset_index()
                        adjusted_display = adjusted_display.groupby(x_col)["ADJUSTED_FORECAST"].sum().reset_index()
                    elif forecast_periods == 52:
                        original_display = original_forecast_display.copy()
                        adjusted_display = adjusted_forecast_display.copy()
                        original_display["QUARTER"] = pd.to_datetime(original_display["WEEK"]).dt.to_period("Q").dt.to_timestamp()
                        adjusted_display["QUARTER"] = pd.to_datetime(adjusted_display["WEEK"]).dt.to_period("Q").dt.to_timestamp()
                        x_col = "QUARTER"
                        x_title = "Quarter"
                        original_display = original_display.groupby(x_col)["FORECAST"].sum().reset_index()
                        adjusted_display = adjusted_display.groupby(x_col)["ADJUSTED_FORECAST"].sum().reset_index()
                    else:
                        original_display = original_forecast_display.copy()
                        adjusted_display = adjusted_forecast_display.copy()
                        x_col = "WEEK"
                        x_title = "Week"
                    
                    col1, col2 = st.columns(2)
                    
                    # --- Prepare display plots and y-axis range before both charts ---
                    original_display_plot = original_display.copy()
                    adjusted_display_plot = adjusted_display.copy()

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

                    with col1:
                        st.subheader("📈 Historical Data")
                        
                        chart_type = st.sidebar.radio(
                            "Chart Type", 
                            ["Line Chart", "Bar Chart"], 
                            index=0, 
                            horizontal=True
                        )

                        historical_fig = go.Figure()

                        if historical_data is not None:
                            if chart_type == "Line Chart":
                                historical_fig.add_trace(go.Scatter(
                                    x=historical_data["WEEK"],
                                    y=historical_data["ACTUAL"],
                                    mode="lines",
                                    name="Historical Actual",
                                    line=dict(color="gray", width=2),
                                    opacity=0.7
                                ))
                            elif chart_type == "Bar Chart":
                                historical_fig.add_trace(go.Bar(
                                    x=historical_data["WEEK"],
                                    y=historical_data["ACTUAL"],
                                    name="Historical Actual",
                                    marker_color="gray",
                                    opacity=0.7
                                ))

                        historical_fig.update_layout(
                            title=f"Historical Actuals - {target_col}",
                            xaxis_title="Week",
                            yaxis_title=target_col,
                            hovermode="x unified",
                            height=500
                        )
                        historical_fig.update_yaxes(tickformat="d", range=[y_min, y_max])

                        st.plotly_chart(historical_fig, use_container_width=True)
                                            
                    with col2:
                        st.subheader("📈 Original vs Adjusted Forecasts")

                        forecast_fig = go.Figure()

                        if chart_type == "Line Chart":
                            forecast_fig.add_trace(go.Scatter(
                                x=original_display_plot[x_col],
                                y=original_display_plot["FORECAST"] if "FORECAST" in original_display_plot else original_display_plot["ADJUSTED_FORECAST"],
                                mode="lines+markers",
                                name="Original Forecast",
                                line=dict(color="blue", width=3),
                                marker=dict(size=6)
                            ))

                            forecast_fig.add_trace(go.Scatter(
                                x=adjusted_display_plot[x_col],
                                y=adjusted_display_plot["ADJUSTED_FORECAST"] if "ADJUSTED_FORECAST" in adjusted_display_plot else adjusted_display_plot["FORECAST"],
                                mode="lines+markers",
                                name="Adjusted Forecast (Lead Time + Buffer)",
                                line=dict(color="red", width=3, dash="dash"),
                                marker=dict(size=6)
                            ))

                        elif chart_type == "Bar Chart":
                            forecast_fig.add_trace(go.Bar(
                                x=original_display_plot[x_col],
                                y=original_display_plot["FORECAST"] if "FORECAST" in original_display_plot else original_display_plot["ADJUSTED_FORECAST"],
                                name="Original Forecast",
                                marker_color="blue",
                                opacity=0.7
                            ))

                            forecast_fig.add_trace(go.Bar(
                                x=adjusted_display_plot[x_col],
                                y=adjusted_display_plot["ADJUSTED_FORECAST"] if "ADJUSTED_FORECAST" in adjusted_display_plot else adjusted_display_plot["FORECAST"],
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

                    anlcol1, anlcol2 = st.columns(2)
                    
                    with anlcol1:
                        st.subheader("📊 Impact Analysis")
                            
                        total_original = original_forecast["FORECAST"].sum()
                        total_adjusted = adjusted_forecast["ADJUSTED_FORECAST"].sum()
                        impact_pct = (total_adjusted - total_original) / total_original * 100 if total_original > 0 else 0

                        metrics_col1, metrics_col2 = st.columns(2)
                            
                        with metrics_col1:
                            st.metric("Original Forecast Total", f"{total_original:,.0f}")
                            st.metric("Lead Time Adjustment", f"{lead_time_weeks} weeks")
                            
                        with metrics_col2:
                            st.metric("Adjusted Forecast Total", f"{total_adjusted:,.0f}"
                                      )
                            st.metric("Total Impact", f"{impact_pct:+.1f}%")
                            
                    with anlcol2:  
                        st.subheader("🛡️ Safety Stock Breakdown")
                        safety_breakdown = pd.DataFrame({
                            "Component": ["Base Forecast", "Stock % Adjustment", "Total Adjusted"],
                            "Value": [
                                total_original,
                                adjusted_forecast["SAFETY_PCT_ADJ"].sum(),
                                total_adjusted
                            ]
                        })
                            
                        fig_bar = px.bar(
                            safety_breakdown,
                            x="Component",
                            y="Value",
                            title="Safety Stock Impact Analysis",
                            color="Component"
                        )
                        fig_bar.update_layout(height=400)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    if inventory_change_table is not None:
                        st.subheader("🔄 Inventory Change for First Forecast Week")
                        st.dataframe(inventory_change_table, use_container_width=True)

                    st.header("📋 Detailed Forecast Data")
                    
                    tab1, tab2 = st.tabs(["Adjusted Forecast", "Comparison"])

                    with tab1:
                        st.subheader("Adjusted Forecast Details")
                        display_adjusted = adjusted_forecast.copy()
                        display_adjusted["Lead Time (weeks)"] = lead_time_weeks
                        if "PRODUCT" not in display_adjusted.columns and "PRODUCT" in df.columns:
                            display_adjusted["PRODUCT"] = df["PRODUCT"].iloc[0]
                        display_adjusted["WEEK"] = display_adjusted["WEEK"].dt.strftime("%Y-%m-%d")
                        display_adjusted = display_adjusted.round(0)
                        display_adjusted = display_adjusted[["WEEK", "PRODUCT", "Lead Time (weeks)", "FORECAST", "SAFETY_PCT_ADJ", "ADJUSTED_FORECAST"]]
                        display_adjusted.columns = ["Week", "Product", "Lead Time (weeks)", "Base Forecast", "Safety % Adj", "Final Forecast"]
                        st.dataframe(display_adjusted, use_container_width=True)

                    with tab2:
                        st.subheader("Side-by-Side Comparison")
                        if "PRODUCT" in original_forecast.columns and "PRODUCT" in adjusted_forecast.columns:
                            comparison_df = pd.merge(
                                original_forecast[["WEEK", "PRODUCT", "FORECAST"]].rename(columns={"FORECAST": "Original"}),
                                adjusted_forecast[["WEEK", "PRODUCT", "ADJUSTED_FORECAST"]].rename(columns={"ADJUSTED_FORECAST": "Adjusted"}),
                                on=["WEEK", "PRODUCT"],
                                how="outer"
                            ).fillna(0)
                            comparison_df["Lead Time (weeks)"] = lead_time_weeks
                            comparison_df["Difference"] = comparison_df["Adjusted"] - comparison_df["Original"]
                            comparison_df["% Change"] = ((comparison_df["Adjusted"] / comparison_df["Original"]) - 1) * 100
                            comparison_df["WEEK"] = pd.to_datetime(comparison_df["WEEK"]).dt.strftime("%Y-%m-%d")
                            comparison_df = comparison_df.round()
                            comparison_df = comparison_df[["WEEK", "PRODUCT", "Lead Time (weeks)", "Original", "Adjusted", "Difference", "% Change"]]
                            comparison_df.columns = ["Week", "Product", "Lead Time (weeks)", "Original", "Adjusted", "Difference", "% Change"]
                        else:
                            comparison_df = pd.merge(
                                original_forecast[["WEEK", "FORECAST"]].rename(columns={"FORECAST": "Original"}),
                                adjusted_forecast[["WEEK", "ADJUSTED_FORECAST"]].rename(columns={"ADJUSTED_FORECAST": "Adjusted"}),
                                on="WEEK",
                                how="outer"
                            ).fillna(0)
                            comparison_df["Lead Time (weeks)"] = lead_time_weeks
                            comparison_df["Difference"] = comparison_df["Adjusted"] - comparison_df["Original"]
                            comparison_df["% Change"] = ((comparison_df["Adjusted"] / comparison_df["Original"]) - 1) * 100
                            comparison_df["WEEK"] = pd.to_datetime(comparison_df["WEEK"]).dt.strftime("%Y-%m-%d")
                            comparison_df = comparison_df.round(2)
                            comparison_df = comparison_df[["WEEK", "Lead Time (weeks)", "Original", "Adjusted", "Difference", "% Change"]]
                            comparison_df.columns = ["Week", "Lead Time (weeks)", "Original", "Adjusted", "Difference", "% Change"]
                        st.dataframe(comparison_df, use_container_width=True)

                    # --- AI Insights Integration ---
                    st.markdown("---")
                    st.subheader("🤖 AI-Powered Insights")

                    # Prepare data for Ollama prompt
                    # Use adjusted_forecast for insights
                    ollama_forecast_data = adjusted_forecast[["WEEK", "ADJUSTED_FORECAST"]].copy()
                    ollama_forecast_data.columns = ["WEEK", "UNITS"]
                    
                    # Assuming single product for simplicity in this prompt generation. 
                    # If multiple products are present, this needs to be handled by iterating or aggregating.
                    if "PRODUCT" in df.columns:
                        product_id_for_ollama = df["PRODUCT"].iloc[0] # Take the first product if multiple
                    else:
                        product_id_for_ollama = "Overall Demand"

                    # Generate prompt for Ollama using the adjusted forecast data
                    ollama_prompt = generate_prompt_ollama(
                        ollama_forecast_data["WEEK"],
                        ollama_forecast_data["UNITS"],
                        product_id_for_ollama,
                        ollama_forecast_data["WEEK"].tolist() # Pass the forecast weeks as future_weeks
                    )

                    if st.button("Generate AI Insights with Ollama"):
                        with st.spinner(f"Generating AI insights using {ollama_model}..."):
                            try:
                                ollama_output = run_ollama_forecast(ollama_prompt, model=ollama_model)
                                # Pass historical units for clipping, if available and relevant
                                historical_units_for_ollama = historical_data["ACTUAL"].values if historical_data is not None else None
                                
                                ollama_forecast_parsed, ollama_markdown_table, ollama_insights = parse_forecast_ollama(
                                    ollama_output, historical_units=historical_units_for_ollama
                                )
                                st.session_state["ollama_insights"] = {
                                    "markdown_table": ollama_markdown_table,
                                    "insights": ollama_insights,
                                    "forecast": ollama_forecast_parsed,
                                    "future_weeks": ollama_forecast_data["WEEK"].tolist(),
                                    "product_id": product_id_for_ollama
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

                        if user_input_chat:
                            st.session_state.chat_history_unified.append({"role": "user", "content": user_input_chat})
                            st.chat_message("user").write(user_input_chat)

                            with st.spinner(f"AI is thinking using {ollama_model}..."):
                                chat_prompt = build_chat_prompt_ollama(
                                    user_input_chat,
                                    ollama_forecast_data, # Use the data sent to Ollama as context
                                    ollama_results["future_weeks"],
                                    ollama_results["forecast"]
                                )
                                response = cached_llm_response(chat_prompt, model=ollama_model)
                            st.session_state.chat_history_unified.append({"role": "assistant", "content": response})
                            st.chat_message("assistant").write(response)

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
            "STORE": ["Store_A", "Store_A", "Store_A"],
            "STORE_TYPE": ["Type_1", "Type_1", "Type_1"],
            "PRODUCT": ["Product_X", "Product_X", "Product_X"],
            "CATEGORY": ["Category_1", "Category_1", "Category_1"],
            "UNITS": [100, 150, 120],
            "SALES": [1000, 1500, 1200],
            "BASE_PRICE": [10, 10, 10],
            "PRICE": [10, 8, 10],
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


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
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Demand Forecasting with Supply Chain Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DemandForecaster:
    def __init__(self):
        self.data = None
        self.forecast_data = None
        self.model = None
        self.scaler = None  # Add scaler attribute
        
        
    def preprocess_data(self, df):
        """Preprocess the uploaded data"""
        # Convert WEEK to datetime if it's not already
        try:
            df['WEEK'] = pd.to_datetime(df['WEEK'])
        except:
            st.error("Error converting WEEK column to datetime. Please ensure WEEK column contains valid dates.")
            return None
            
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Handle categorical missing values
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('Unknown')
        
        # Sort by week
        df = df.sort_values('WEEK')
        
        return df
    
    def create_forecast_features(self, df, target_col):
        """Create features for forecasting"""
        # Aggregate data by week if multiple records per week
        weekly_data = df.groupby('WEEK').agg({
            target_col: 'sum',
            'BASE_PRICE': 'mean',
            'PRICE': 'mean',
            'FEATURE': 'max',
            'DISPLAY': 'max',
            'INVENTORY': 'sum',
            'VISITS': 'sum'
        }).reset_index()
        
        # Create lag features
        for lag in [1, 2, 4, 8]:
            weekly_data[f'{target_col}_lag_{lag}'] = weekly_data[target_col].shift(lag)
        
        # Create rolling averages
        for window in [4, 8, 12]:
            weekly_data[f'{target_col}_rolling_{window}'] = weekly_data[target_col].rolling(window=window).mean()
        
        # Create trend and seasonal features
        weekly_data['week_number'] = weekly_data['WEEK'].dt.isocalendar().week
        weekly_data['month'] = weekly_data['WEEK'].dt.month
        weekly_data['quarter'] = weekly_data['WEEK'].dt.quarter
        
        # Price elasticity feature
        weekly_data['price_ratio'] = weekly_data['PRICE'] / weekly_data['BASE_PRICE']
        weekly_data['price_ratio'] = weekly_data['price_ratio'].fillna(1)
        
        return weekly_data
    
    def train_forecast_model(self, df, target_col, forecast_periods=12, model_type='Random Forest'):
        """Train forecasting model and generate predictions"""
        # Prepare features
        feature_columns = [col for col in df.columns if col not in ['WEEK', target_col] and not col.startswith(target_col)]
        
        # Remove rows with NaN values (due to lag features)
        clean_df = df.dropna()
        
        if len(clean_df) < 10:
            st.error("Not enough data points for forecasting. Need at least 10 complete records.")
            return None, None
        
        X = clean_df[feature_columns]
        y = clean_df[target_col]
        
        # Split data (use last 20% for validation)
        split_idx = int(len(clean_df) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        if model_type == 'Linear Regression':
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            self.model = LinearRegression()
            self.model.fit(X_train_scaled, y_train)
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

        # Generate forecasts
        forecast_dates = pd.date_range(start=df['WEEK'].max() + pd.Timedelta(weeks=1), 
                                     periods=forecast_periods, freq='W')
        
        # Create future features (simplified - using last known values and trends)
        last_row = clean_df.iloc[-1:].copy()
        forecasts = []
        
        for i, date in enumerate(forecast_dates):
            future_row = last_row.copy()
            future_row['WEEK'] = date
            future_row['week_number'] = date.isocalendar()[1]
            future_row['month'] = date.month
            future_row['quarter'] = date.quarter
            
            # Simple trend continuation for lag features
            if i == 0:
                for lag in [1, 2, 4, 8]:
                    if f'{target_col}_lag_{lag}' in future_row.columns:
                        future_row[f'{target_col}_lag_{lag}'] = clean_df[target_col].iloc[-lag] if len(clean_df) >= lag else clean_df[target_col].iloc[-1]
            
            X_future = future_row[feature_columns]
            if model_type == 'Linear Regression':
                X_future = self.scaler.transform(X_future)
            pred = self.model.predict(X_future)[0]
            forecasts.append({
                'WEEK': date,
                'FORECAST': max(0, pred),  # Ensure non-negative forecasts
                'TYPE': 'Original'
            })
        
        forecast_df = pd.DataFrame(forecasts)
        
        # Historical data for comparison
        if model_type == 'Linear Regression':
            historical_pred = self.model.predict(self.scaler.transform(X))
        else:
            historical_pred = self.model.predict(X)
        historical_df = pd.DataFrame({
            'WEEK': clean_df['WEEK'],
            'ACTUAL': clean_df[target_col],
            'FORECAST': historical_pred,
            'TYPE': 'Historical'
        })
        return forecast_df, historical_df
    
    def apply_supply_chain_adjustments(self, forecast_df, lead_time_weeks, safety_pct, safety_qty):
        """Apply lead time and Safety Stock adjustments to forecasts"""
        adjusted_forecast = forecast_df.copy()
        
        # Apply lead time adjustment (shift forecast forward)
        if lead_time_weeks > 0:
            adjusted_forecast['WEEK'] = adjusted_forecast['WEEK'] + pd.Timedelta(weeks=lead_time_weeks)
        
        # Apply Safety Stock adjustments
        adjusted_forecast['SAFETY_PCT_ADJ'] = adjusted_forecast['FORECAST'] * (safety_pct / 100)
        adjusted_forecast['TOTAL_SAFETY'] = adjusted_forecast['SAFETY_PCT_ADJ'] + safety_qty
        adjusted_forecast['ADJUSTED_FORECAST'] = adjusted_forecast['FORECAST'] + adjusted_forecast['TOTAL_SAFETY']
        
        # Ensure non-negative values
        adjusted_forecast['ADJUSTED_FORECAST'] = adjusted_forecast['ADJUSTED_FORECAST'].clip(lower=0)
        
        adjusted_forecast['TYPE'] = 'Adjusted'
        
        return adjusted_forecast

def main():
    st.title("📈 Demand Forecasting with Supply Chain Optimization")
    st.markdown("### Upload your retail demand data and optimize forecasts with Lead Time and Safety Stock parameters")
    
    # Sidebar for file upload and parameters
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and validate data
            df = pd.read_csv(uploaded_file)
            
            # --- Feature Explanation Table ---
            st.subheader("📝 Feature Descriptions")
            feature_descriptions = {
                "WEEK": "The week of the observation (date or week number).",
                "STORE": "Store identifier or name.",
                "STORE_TYPE": "Type/category of the store.",
                "PRODUCT": "Product identifier or name.",
                "CATEGORY": "Product category.",
                "UNITS": "Units sold during the week.",
                "SALES": "Sales revenue for the week.",
                "BASE_PRICE": "Standard price of the product.",
                "PRICE": "Actual selling price during the week.",
                "INVENTORY": "Inventory level at the start/end of the week.",
                "FEATURE": "Whether the product was featured/promoted (binary/flag).",
                "DISPLAY": "Whether the product was on display (binary/flag).",
                "VISITS": "Number of customer visits."
            }
            feature_table = pd.DataFrame({
                "Feature": df.columns,
                "Description": [feature_descriptions.get(col, "No description available.") for col in df.columns]
            })
            st.dataframe(feature_table, use_container_width=True)
            # --- End Feature Explanation Table ---

            # Validate required columns
            required_columns = ['WEEK', 'UNITS', 'SALES']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                st.stop()
            
            st.sidebar.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Initialize forecaster
            forecaster = DemandForecaster()
            
            # Preprocess data
            df = forecaster.preprocess_data(df)
            if df is None:
                st.stop()
            
            # Sidebar filters
            st.sidebar.header("🎯 Forecast Configuration")
            
            # Add model selection to sidebar
            model_type = st.sidebar.selectbox("Forecast Model", ["Random Forest", "Linear Regression"])
            st.sidebar.markdown("Select the model type for forecasting. Random Forest is generally more robust, while Linear Regression is simpler and faster.")

            # Target variable selection
            target_options = ['UNITS', 'SALES', 'INVENTORY']
            target_col = st.sidebar.selectbox("Select Target Variable", target_options)
            
            # Forecasting horizon
            forecast_periods = st.sidebar.slider("Forecast Periods (weeks)", 4, 52, 4)
            
            # Data filtering options
            st.sidebar.header("📊 Data Filters")

                        # --- Dynamic Multi-Select Filters ---
            filter_options = []
            if 'PRODUCT' in df.columns:
                filter_options.append('Product')
            if 'MANUFACTURER' in df.columns:
                filter_options.append('Manufacturer')
            if 'BRAND' in df.columns:
                filter_options.append('Brand')
            if 'CATEGORY' in df.columns:
                filter_options.append('Category')
            if 'SUBCATEGORY' in df.columns:
                filter_options.append('Sub-Category')
            if 'SIZE' in df.columns:
                filter_options.append('Size')

            selected_filters = st.sidebar.multiselect(
                "Select Filters to Apply",
                filter_options,
                default=[]
            )

            # Product filter
            if 'Product' in selected_filters and 'PRODUCT' in df.columns:
                products = [str(int(p)) if isinstance(p, (int, float)) and float(p).is_integer() else str(p) for p in sorted(df['PRODUCT'].unique())]
                selected_products = st.sidebar.multiselect("Select Product(s)", products, default=products)
                if selected_products:
                    df = df[df['PRODUCT'].astype(str).isin(selected_products)]

            # Manufacturer filter
            if 'Manufacturer' in selected_filters and 'MANUFACTURER' in df.columns:
                manufacturers = sorted(df['MANUFACTURER'].astype(str).unique())
                selected_manufacturers = st.sidebar.multiselect("Select Manufacturer(s)", manufacturers, default=manufacturers)
                if selected_manufacturers:
                    df = df[df['MANUFACTURER'].astype(str).isin(selected_manufacturers)]

            # Brand filter
            if 'Brand' in selected_filters and 'BRAND' in df.columns:
                brands = sorted(df['BRAND'].astype(str).unique())
                selected_brands = st.sidebar.multiselect("Select Brand(s)", brands, default=brands)
                if selected_brands:
                    df = df[df['BRAND'].astype(str).isin(selected_brands)]

            # Category filter
            if 'Category' in selected_filters and 'CATEGORY' in df.columns:
                categories = sorted(df['CATEGORY'].astype(str).unique())
                selected_categories = st.sidebar.multiselect("Select Category(s)", categories, default=categories)
                if selected_categories:
                    df = df[df['CATEGORY'].astype(str).isin(selected_categories)]

            # Sub-Category filter
            if 'Sub-Category' in selected_filters and 'SUBCATEGORY' in df.columns:
                sub_categories = sorted(df['SUBCATEGORY'].astype(str).unique())
                selected_sub_categories = st.sidebar.multiselect("Select Sub-Category(s)", sub_categories, default=sub_categories)
                if selected_sub_categories:
                    df = df[df['SUBCATEGORY'].astype(str).isin(selected_sub_categories)]

            # Size filter
            if 'Size' in selected_filters and 'SIZE' in df.columns:
                sizes = sorted(df['SIZE'].astype(str).unique())
                selected_sizes = st.sidebar.multiselect("Select Size(s)", sizes, default=sizes)
                if selected_sizes:
                    df = df[df['SIZE'].astype(str).isin(selected_sizes)]
                                
            # Supply Chain Parameters
            st.sidebar.header("🚛 Supply Chain Parameters")
            lead_time_weeks = st.sidebar.slider("Lead Time (weeks)", 0, 12, 0, 
                                              help="Time between order placement and delivery")

            safety_pct = st.sidebar.slider("Safety Stock (%)", 0, 100, 0,
                                         help="Percentage adjustment to forecasted demand")

            safety_qty = st.sidebar.number_input("Safety Stock (Quantity)", 0, 1000000, 0,
                                               help="Fixed quantity adjustment to Safety Stock")
            
            # Generate forecasts
            st.header("🔮 Demand Forecasting Results")
            
            with st.spinner("Generating forecasts..."):
                # Prepare data for forecasting
                forecast_data = forecaster.create_forecast_features(df, target_col)
                
                # Pass model_type to train_forecast_model
                original_forecast, historical_data = forecaster.train_forecast_model(
                    forecast_data, target_col, forecast_periods, model_type
                )
                
                if original_forecast is not None:
                    # Apply supply chain adjustments
                    adjusted_forecast = forecaster.apply_supply_chain_adjustments(
                        original_forecast, lead_time_weeks, safety_pct, safety_qty
                    )
                    
                    # Create visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 Historical Data")
                        
                        historicalFig = go.Figure()
                        
                        # Add historical actual data if available
                        if historical_data is not None:
                            historicalFig.add_trace(go.Scatter(
                                x=historical_data['WEEK'],
                                y=historical_data['ACTUAL'],
                                mode='lines',
                                name='Historical Actual',
                                line=dict(color='gray', width=2),
                                opacity=0.7
                            ))

                        historicalFig.update_layout(
                            title=f"Historical Actuals - {target_col}",
                            xaxis_title="Week",
                            yaxis_title=target_col,
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(historicalFig, use_container_width=True)
                    
                    with col2:
                        st.subheader("📈 Original vs Adjusted Forecasts")
                        
                        forecastFig = go.Figure()
                        
                        # Add original forecast
                        forecastFig.add_trace(go.Scatter(
                            x=original_forecast['WEEK'],
                            y=original_forecast['FORECAST'],
                            mode='lines+markers',
                            name='Original Forecast',
                            line=dict(color='blue', width=3),
                            marker=dict(size=6)
                        ))
                        
                        # Add adjusted forecast
                        forecastFig.add_trace(go.Scatter(
                            x=adjusted_forecast['WEEK'],
                            y=adjusted_forecast['ADJUSTED_FORECAST'],
                            mode='lines+markers',
                            name='Adjusted Forecast (Lead Time + Buffer)',
                            line=dict(color='red', width=3, dash='dash'),
                            marker=dict(size=6)
                        ))

                        forecastFig.update_layout(
                            title=f"Demand Forecast Comparison - {target_col}",
                            xaxis_title="Week",
                            yaxis_title=target_col,
                            hovermode='x unified',
                            height=500
                        )

                        st.plotly_chart(forecastFig, use_container_width=True)

                    anlcol1, anlcol2 = st.columns(2)
                    
                    with anlcol1:
                        st.subheader("📊 Impact Analysis")
                            
                        # Calculate impact metrics
                        total_original = original_forecast['FORECAST'].sum()
                        total_adjusted = adjusted_forecast['ADJUSTED_FORECAST'].sum()
                        impact_qty = total_adjusted - total_original
                        impact_pct = (impact_qty / total_original) * 100 if total_original > 0 else 0
                            
                            # Display metrics
                        metrics_col1, metrics_col2 = st.columns(2)
                            
                        with metrics_col1:
                            st.metric("Original Forecast Total", f"{total_original:,.0f}")
                            st.metric("Lead Time Adjustment", f"{lead_time_weeks} weeks")
                            
                        with metrics_col2:
                            st.metric("Adjusted Forecast Total", f"{total_adjusted:,.0f}", 
                                    delta=f"{impact_qty:+,.0f}")
                            st.metric("Total Impact", f"{impact_pct:+.1f}%")
                            
                    with anlcol2:  
                        # Safety Stock breakdown
                        st.subheader("🛡️ Safety Stock Breakdown")
                        safety_breakdown = pd.DataFrame({
                            'Component': ['Base Forecast', 'Stock % Adjustment', 'Stock Qty Adjustment', 'Total Adjusted'],
                            'Value': [
                                total_original,
                                adjusted_forecast['SAFETY_PCT_ADJ'].sum(),
                                safety_qty * len(adjusted_forecast),
                                total_adjusted
                            ]
                        })
                            
                        fig_bar = px.bar(
                            safety_breakdown,
                            x='Component',
                            y='Value',
                            title="Safety Stock Impact Analysis",
                            color='Component'
                        )
                        fig_bar.update_layout(height=400)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                                        # Detailed forecast tables
                    st.header("📋 Detailed Forecast Data")
                    
                    tab1, tab2 = st.tabs(["Adjusted Forecast", "Comparison"])
                    
                    with tab1:
                        st.subheader("Adjusted Forecast Details")
                        display_adjusted = adjusted_forecast[['WEEK', 'FORECAST', 'SAFETY_PCT_ADJ', 'TOTAL_SAFETY', 'ADJUSTED_FORECAST']].copy()
                        display_adjusted['WEEK'] = display_adjusted['WEEK'].dt.strftime('%Y-%m-%d')
                        display_adjusted = display_adjusted.round(0)
                        display_adjusted.columns = ['Week', 'Base Forecast', 'Safety % Adj', 'Total Safety', 'Final Forecast']
                        st.dataframe(display_adjusted, use_container_width=True)
                    
                    with tab2:
                        st.subheader("Side-by-Side Comparison")
                        comparison_df = pd.merge(
                            original_forecast[['WEEK', 'FORECAST']].rename(columns={'FORECAST': 'Original'}),
                            adjusted_forecast[['WEEK', 'ADJUSTED_FORECAST']].rename(columns={'ADJUSTED_FORECAST': 'Adjusted'}),
                            on='WEEK',
                            how='outer'
                        ).fillna(0)
                        comparison_df['Difference'] = comparison_df['Adjusted'] - comparison_df['Original']
                        comparison_df['% Change'] = ((comparison_df['Adjusted'] / comparison_df['Original']) - 1) * 100
                        comparison_df['WEEK'] = comparison_df['WEEK'].dt.strftime('%Y-%m-%d')
                        comparison_df = comparison_df.round(2)
                        st.dataframe(comparison_df, use_container_width=True)

                        # Model performance metrics
                    if historical_data is not None:
                        st.header("🎯 Model Performance")
                        mae = mean_absolute_error(historical_data['ACTUAL'], historical_data['FORECAST'])
                        rmse = np.sqrt(mean_squared_error(historical_data['ACTUAL'], historical_data['FORECAST']))
                            
                        perf_col1, perf_col2, perf_col3 = st.columns(3)
                        with perf_col1:
                            st.metric("Mean Absolute Error", f"{mae:.2f}")
                        with perf_col2:
                            st.metric("Root Mean Square Error", f"{rmse:.2f}")
                        with perf_col3:
                            mape = np.mean(np.abs((historical_data['ACTUAL'] - historical_data['FORECAST']) / historical_data['ACTUAL'])) * 100
                            st.metric("Mean Absolute % Error", f"{mape:.1f}%")
                
                else:
                    st.error("Unable to generate forecasts. Please check your data quality and try again.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file contains the required columns: WEEK, UNITS, SALES")
    
    else:
        # Display sample data format
        st.info("👆 Please upload a CSV file to get started")
        st.subheader("📋 Expected Data Format")
        
        sample_data = pd.DataFrame({
            'WEEK': ['2024-01-01', '2024-01-08', '2024-01-15'],
            'STORE': ['Store_A', 'Store_A', 'Store_A'],
            'STORE_TYPE': ['Type_1', 'Type_1', 'Type_1'],
            'PRODUCT': ['Product_X', 'Product_X', 'Product_X'],
            'CATEGORY': ['Category_1', 'Category_1', 'Category_1'],
            'UNITS': [100, 150, 120],
            'SALES': [1000, 1500, 1200],
            'BASE_PRICE': [10, 10, 10],
            'PRICE': [10, 8, 10],
            'INVENTORY': [500, 400, 450]
        })
        
        st.dataframe(sample_data)
        
        st.markdown("""
        ### Key Features:
        - **Lead Time Adjustment**: Shifts forecast timeline to account for supply chain delays
        - **Safety Stock %**: Applies percentage-based safety stock to forecasted demand
        - **Safety Stock Quantity**: Adds fixed quantity buffer to inventory planning
        - **Interactive Visualizations**: Real-time updates as you adjust parameters
        - **Multi-level Filtering**: Analyze by store, product, or category
        - **Performance Metrics**: Evaluate forecast accuracy with standard metrics
        """)

if __name__ == "__main__":
    main()
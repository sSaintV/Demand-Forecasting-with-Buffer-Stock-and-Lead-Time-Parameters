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
    
    def train_forecast_model(self, df, target_col, forecast_periods=12):
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
            
            pred = self.model.predict(future_row[feature_columns])[0]
            forecasts.append({
                'WEEK': date,
                'FORECAST': max(0, pred),  # Ensure non-negative forecasts
                'TYPE': 'Original'
            })
        
        forecast_df = pd.DataFrame(forecasts)
        
        # Historical predictions for comparison
        historical_pred = self.model.predict(X)
        historical_df = pd.DataFrame({
            'WEEK': clean_df['WEEK'],
            'ACTUAL': clean_df[target_col],
            'FORECAST': historical_pred,
            'TYPE': 'Historical'
        })
        
        return forecast_df, historical_df
    
    def apply_supply_chain_adjustments(self, forecast_df, lead_time_weeks, buffer_pct, buffer_qty):
        """Apply lead time and buffer stock adjustments to forecasts"""
        adjusted_forecast = forecast_df.copy()
        
        # Apply lead time adjustment (shift forecast forward)
        if lead_time_weeks > 0:
            adjusted_forecast['WEEK'] = adjusted_forecast['WEEK'] + pd.Timedelta(weeks=lead_time_weeks)
        
        # Apply buffer stock adjustments
        adjusted_forecast['BUFFER_PCT_ADJ'] = adjusted_forecast['FORECAST'] * (buffer_pct / 100)
        adjusted_forecast['TOTAL_BUFFER'] = adjusted_forecast['BUFFER_PCT_ADJ'] + buffer_qty
        adjusted_forecast['ADJUSTED_FORECAST'] = adjusted_forecast['FORECAST'] + adjusted_forecast['TOTAL_BUFFER']
        
        # Ensure non-negative values
        adjusted_forecast['ADJUSTED_FORECAST'] = adjusted_forecast['ADJUSTED_FORECAST'].clip(lower=0)
        
        adjusted_forecast['TYPE'] = 'Adjusted'
        
        return adjusted_forecast

def main():
    st.title("📈 Demand Forecasting with Supply Chain Optimization")
    st.markdown("### Upload your retail demand data and optimize forecasts with Lead Time and Buffer Stock parameters")
    
    # Sidebar for file upload and parameters
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and validate data
            df = pd.read_csv(uploaded_file)
            
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
            
            # Target variable selection
            target_options = ['UNITS', 'SALES', 'INVENTORY']
            target_col = st.sidebar.selectbox("Select Target Variable", target_options)
            
            # Forecasting horizon
            forecast_periods = st.sidebar.slider("Forecast Periods (weeks)", 4, 52, 4)
            
            # Data filtering options
            st.sidebar.header("📊 Data Filters")
            
            # Store filters
            if 'STORE' in df.columns:
                stores = ['All'] + list(df['STORE'].unique())
                # Remove decimal points from store names if they are numbers
                stores = ['All'] + [str(int(s)) if isinstance(s, (int, float)) and float(s).is_integer() else str(s) for s in df['STORE'].unique()]
                selected_store = st.sidebar.selectbox("Select Store", stores)
                if selected_store != 'All':
                    df = df[df['STORE'].astype(str) == selected_store]

            # Product filters
            if 'PRODUCT' in df.columns:
                products = ['All'] + sorted(df['PRODUCT'].unique())
                # Remove decimal points from product names if they are numbers
                products = ['All'] + [str(int(p)) if isinstance(p, (int, float)) and float(p).is_integer() else str(p) for p in sorted(df['PRODUCT'].unique())]
                selected_product = st.sidebar.selectbox("Select Product", products)
                if selected_product != 'All':
                    df = df[df['PRODUCT'].astype(str) == selected_product]
            
            # Supply Chain Parameters
            st.sidebar.header("🚛 Supply Chain Parameters")
            lead_time_weeks = st.sidebar.slider("Lead Time (weeks)", 0, 12, 0, 
                                              help="Time between order placement and delivery")
            
            buffer_pct = st.sidebar.slider("Buffer Stock (%)", 0, 100, 0,
                                         help="Percentage adjustment to forecasted demand")
            
            buffer_qty = st.sidebar.number_input("Buffer Stock (Quantity)", 0, 1000000, 0,
                                               help="Fixed quantity adjustment to buffer stock")
            
            # Generate forecasts
            st.header("🔮 Demand Forecasting Results")
            
            with st.spinner("Generating forecasts..."):
                # Prepare data for forecasting
                forecast_data = forecaster.create_forecast_features(df, target_col)
                
                # Train model and generate forecasts
                original_forecast, historical_data = forecaster.train_forecast_model(
                    forecast_data, target_col, forecast_periods
                )
                
                if original_forecast is not None:
                    # Apply supply chain adjustments
                    adjusted_forecast = forecaster.apply_supply_chain_adjustments(
                        original_forecast, lead_time_weeks, buffer_pct, buffer_qty
                    )
                    
                    # Create visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 Historical Forecasts")
                        
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
                        
                        # Buffer stock breakdown
                        st.subheader("🛡️ Buffer Stock Breakdown")
                        buffer_breakdown = pd.DataFrame({
                            'Component': ['Base Forecast', 'Buffer % Adjustment', 'Buffer Qty Adjustment', 'Total Adjusted'],
                            'Value': [
                                total_original,
                                adjusted_forecast['BUFFER_PCT_ADJ'].sum(),
                                buffer_qty * len(adjusted_forecast),
                                total_adjusted
                            ]
                        })
                        
                        fig_bar = px.bar(
                            buffer_breakdown, 
                            x='Component', 
                            y='Value',
                            title="Buffer Stock Impact Analysis",
                            color='Component'
                        )
                        fig_bar.update_layout(height=400)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Detailed forecast tables
                    st.header("📋 Detailed Forecast Data")
                    
                    tab1, tab2 = st.tabs(["Adjusted Forecast", "Comparison"])
                    
                    with tab1:
                        st.subheader("Adjusted Forecast Details")
                        display_adjusted = adjusted_forecast[['WEEK', 'FORECAST', 'BUFFER_PCT_ADJ', 'TOTAL_BUFFER', 'ADJUSTED_FORECAST']].copy()
                        display_adjusted['WEEK'] = display_adjusted['WEEK'].dt.strftime('%Y-%m-%d')
                        display_adjusted = display_adjusted.round(0)
                        display_adjusted.columns = ['Week', 'Base Forecast', 'Buffer % Adj', 'Total Buffer', 'Final Forecast']
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
        - **Buffer Stock %**: Applies percentage-based safety stock to forecasted demand
        - **Buffer Stock Quantity**: Adds fixed quantity buffer to inventory planning
        - **Interactive Visualizations**: Real-time updates as you adjust parameters
        - **Multi-level Filtering**: Analyze by store, product, or category
        - **Performance Metrics**: Evaluate forecast accuracy with standard metrics
        """)

if __name__ == "__main__":
    main()
# 📈 Demand Forecasting with Buffer Stock and Lead Time Parameters

A web-based demand forecasting application built with **Python** and **Streamlit**, designed specifically for supply chain and inventory planners. It integrates time-series forecasting with supply chain parameters such as **lead time** and **buffer stock**, enabling users to make more reliable inventory decisions in uncertain demand environments.

## 🔍 Project Overview

This application empowers data scientists, ML engineers, business analysts, and developers to:

- Forecast retail demand based on historical sales and inventory data.
- Incorporate **supply chain constraints** (lead time delays) and **risk mitigation strategies** (buffer stock).
- Visualize the impact of adjustments and optimize safety stock levels.
- Interactively explore product/store-level forecasts and model performance.

The app provides an **interactive interface** for uploading datasets, configuring forecasting settings, visualizing predictions, and analyzing performance metrics — all without needing to write a single line of code.

---

## ✅ Key Features

### 📊 Demand Forecasting
- **Time-series forecasting** using a Random Forest Regressor.
- Uses lag features, rolling averages, and seasonal indicators.
- Incorporates price elasticity and promotional factors (`PRICE`, `FEATURE`, `DISPLAY`).

### 🛠️ Supply Chain Adjustments
- **Lead Time Adjustment**: Shifts forecast horizon forward by X weeks.
- **Buffer Stock %**: Applies a percentage-based increase to forecasted demand.
- **Buffer Stock Quantity**: Adds a fixed quantity buffer to each forecast period.

### 📈 Visual Analytics
- Interactive **Plotly charts** for comparing original and adjusted forecasts.
- **Bar chart breakdown** of base vs. adjusted forecasts.
- Side-by-side data tables for validation and export.

### 📉 Model Performance
- Evaluates predictions using:
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Percentage Error (MAPE)**

### 🧭 Business-Friendly Filters
- Filter data by `STORE` and `PRODUCT` (if columns exist).
- Switch between forecasting `UNITS` or `SALES`.

---

## 📁 Data Input Requirements

Upload a CSV file containing weekly retail data. Minimum required columns:

| Column       | Description                            |
|--------------|----------------------------------------|
| `WEEK`       | Weekly date (e.g. `2024-01-01`)         |
| `UNITS`      | Quantity sold per week                 |
| `SALES`      | Sales value per week                   |

Optional columns for enhanced modeling:

- `BASE_PRICE`, `PRICE`: Used to calculate price elasticity.
- `FEATURE`, `DISPLAY`: Promotional indicators.
- `INVENTORY`, `VISITS`: Additional demand signals.
- `STORE`, `PRODUCT`, `CATEGORY`: Enable filtering.

---

## ⚙️ Installation Guide

### 🔧 System Requirements

- Python 3.7 or higher
- pip (Python package manager)

### 📦 Dependencies

Install required packages:

```bash
pip install streamlit pandas numpy plotly scikit-learn
````

### 🚀 Running the Application

After installing the dependencies:

```bash
streamlit run demand_forecasting_app.py
```

This will launch the Streamlit web app in your default browser.

---

## 👥 Audience-Specific Usage

### 👨‍🔬 Data Scientists & ML Engineers

* Understand how feature engineering (lag, rolling, seasonality) feeds into a Random Forest model.
* Access model diagnostics (MAE, RMSE, MAPE).
* Extend the model with alternate algorithms (e.g., XGBoost).

### 📊 Business Analysts

* Use the no-code interface to upload data and set lead time/buffer levels.
* Visualize how operational changes affect inventory targets.
* Export forecast tables for reporting or ERP integration.

### 👨‍💻 General Developers

* Easily customize or extend the app using modular Python class `DemandForecaster`.
* Contribute to UI/UX or integrate with ERP APIs for real-time updates.

---

## 🧮 Technical Details

### Forecasting Methodology

* **Model**: RandomForestRegressor from `scikit-learn`
* **Features**:

  * Lag variables (`UNITS_lag_1`, `UNITS_lag_2`, etc.)
  * Rolling means (4-, 8-, 12-week windows)
  * Temporal signals: week number, month, quarter
  * Price elasticity: `PRICE / BASE_PRICE`
* **Training Strategy**:

  * 80/20 train-validation split
  * Forecast horizon configurable (4 to 26 weeks)

### Supply Chain Calculations

#### Lead Time Adjustment:

```python
adjusted_forecast['WEEK'] += pd.Timedelta(weeks=lead_time_weeks)
```

#### Buffer Stock Adjustment:

```python
adjusted_forecast['BUFFER_PCT_ADJ'] = FORECAST * (buffer_pct / 100)
adjusted_forecast['TOTAL_BUFFER'] = BUFFER_PCT_ADJ + buffer_qty
adjusted_forecast['ADJUSTED_FORECAST'] = FORECAST + TOTAL_BUFFER
```

---

## 🧪 Sample Data Format

Here’s an example of the minimum structure:

```csv
WEEK,STORE,PRODUCT,UNITS,SALES,BASE_PRICE,PRICE,FEATURE,DISPLAY,INVENTORY
2024-01-01,Store_A,Product_X,100,1000,10,10,1,1,500
2024-01-08,Store_A,Product_X,150,1500,10,8,1,0,400
2024-01-15,Store_A,Product_X,120,1200,10,10,0,0,450
```

---

## 🛠️ Troubleshooting

| Issue                        | Solution                                                                             |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| **App won’t start**          | Ensure Python ≥ 3.7 is installed. Run via `streamlit run demand_forecasting_app.py`. |
| **Missing columns**          | Check CSV contains `WEEK`, `UNITS`, `SALES`. Others are optional but recommended.    |
| **Forecasts not generated**  | Ensure at least 10 rows of clean data with no missing `UNITS`/`SALES`.               |
| **Charts not rendering**     | Ensure browser supports JavaScript. Try in Chrome or Firefox.                        |
| **Inconsistent date format** | Ensure `WEEK` column is in a recognizable date format (`YYYY-MM-DD`).                |

---

## 🤝 Contributing

Contributions are welcome! If you'd like to:

* Add new forecasting algorithms (e.g., Prophet, XGBoost)
* Improve visualizations
* Extend CSV output or ERP integration

Please fork the repository and submit a pull request.

---

## 📬 Contact

For questions, feedback, or enterprise support:

* GitHub: [@sSaintV](https://github.com/sSaintV)
* Email: *Add contact email if applicable*

---

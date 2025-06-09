# sales-forcasting-cloudcredit
Time series sales forecasting using Streamlit
This project is a Streamlit web application that allows users to upload historical sales data and forecast future sales using multiple time series and machine learning models.

🔍 Features
Upload CSV file with sales data

Compare ARIMA, Exponential Smoothing, and Random Forest models

Interactive visualizations of:

Forecast vs actual

Model performance metrics (MAE, RMSE)

Clean and simple Streamlit UI

🛠️ Technologies Used
streamlit – for the web app UI

pandas, numpy – for data manipulation

matplotlib – for plotting

statsmodels – for ARIMA and ETS models

scikit-learn – for Random Forest and evaluation metrics

📂 File Structure
bash
Copy
Edit
sales_forecast_app/
├── app.py               # Streamlit application
├── requirements.txt     # Required Python packages
└── sales_sample.csv     # Sample dataset for testing

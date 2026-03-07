import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Time Series
from statsmodels.tsa.arima.model import ARIMA

# =========================
# Load Data
# =========================

df = pd.read_csv("amazon_sales_dataset.csv")

df.columns = df.columns.str.lower().str.strip()

df['order_date'] = pd.to_datetime(df['order_date'])

df["total_revenue"] = (
    df["price"] *
    df["quantity_sold"] *
    (1 - df["discount_percent"] / 100)
)

# =========================
# Page Title
# =========================

st.title("📊 E-Commerce Revenue Analytics Dashboard")

st.sidebar.header("Navigation")

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Project Overview",
        "Data Overview",
        "EDA",
        "Revenue Prediction",
        "Revenue Forecast"
    ]
)

# =========================
# PROJECT OVERVIEW
# =========================

if menu == "Project Overview":

    st.header("📌 Business Background")

    st.write("""
Perkembangan e-commerce mendorong perusahaan untuk tidak hanya meningkatkan jumlah transaksi,
tetapi juga mengoptimalkan revenue yang dihasilkan dari setiap transaksi.

Dengan banyaknya variasi harga produk, diskon, serta perbedaan perilaku customer di berbagai wilayah,
perusahaan membutuhkan analisis data untuk memahami faktor yang mempengaruhi revenue.
""")

    st.header("🎯 Business Objectives")

    st.write("""
1. Mengidentifikasi faktor yang mempengaruhi revenue transaksi  
2. Memprediksi revenue transaksi menggunakan Machine Learning  
3. Memprediksi revenue masa depan menggunakan Time Series Forecasting  
""")

    st.header("⚙️ Methodology")

    st.write("""
1. Exploratory Data Analysis (EDA)
2. Machine Learning (Revenue Prediction)
3. Time Series Forecasting (Revenue Forecast)
4. Model Evaluation
""")

# =========================
# DATA OVERVIEW
# =========================

elif menu == "Data Overview":

    st.header("Dataset Overview")

    st.write("Shape Dataset:", df.shape)

    st.subheader("Preview Data")

    st.dataframe(df.head())

    st.subheader("Statistical Summary")

    st.write(df.describe())

# =========================
# EDA
# =========================

elif menu == "EDA":

    st.header("Exploratory Data Analysis")

    # Revenue Distribution
    st.subheader("Revenue Distribution")

    fig, ax = plt.subplots()

    sns.histplot(df["total_revenue"], bins=50, kde=True, ax=ax)

    st.pyplot(fig)

    # Category Revenue
    st.subheader("Revenue by Category")

    category_revenue = (
        df.groupby("product_category")["total_revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()

    category_revenue.plot(kind="bar", ax=ax)

    st.pyplot(fig)

    # Quantity vs Revenue
    st.subheader("Quantity Sold vs Revenue")

    fig, ax = plt.subplots()

    sns.regplot(
        x="quantity_sold",
        y="total_revenue",
        data=df,
        scatter_kws={"alpha":0.4},
        ax=ax
    )

    st.pyplot(fig)

    # Discount Analysis
    st.subheader("Discount Impact")

    df["discount_flag"] = (df["discount_percent"] > 0).astype(int)

    fig, ax = plt.subplots()

    sns.boxplot(x="discount_flag", y="total_revenue", data=df, ax=ax)

    ax.set_xticklabels(["No Discount","Discount"])

    st.pyplot(fig)

    # Regional Revenue
    st.subheader("Revenue by Region")

    region_revenue = (
        df.groupby("customer_region")["total_revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()

    region_revenue.plot(kind="bar", ax=ax)

    st.pyplot(fig)

    # Monthly Trend
    st.subheader("Monthly Revenue Trend")

    monthly_revenue = (
        df.set_index("order_date")
        .resample("M")["total_revenue"]
        .sum()
    )

    fig, ax = plt.subplots()

    monthly_revenue.plot(ax=ax)

    st.pyplot(fig)

# =========================
# REVENUE PREDICTION
# =========================

elif menu == "Revenue Prediction":

    st.header("Revenue Prediction (Machine Learning)")

    st.info("""
Model menggunakan Random Forest Regressor untuk memprediksi revenue transaksi
berdasarkan price, quantity_sold, dan discount_percent.
""")

    X = df[["price", "quantity_sold", "discount_percent"]]

    y = df["total_revenue"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Model
    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")

    st.write("MAE:", mae)

    st.write("RMSE:", rmse)

    st.write("R²:", r2)

    # Feature Importance
    st.subheader("Feature Importance")

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    )

    st.bar_chart(importance)

    # Prediction Simulation
    st.subheader("Revenue Prediction Simulation")

    price = st.number_input("Price", value=100)

    quantity = st.number_input("Quantity Sold", value=10)

    discount = st.slider("Discount (%)", 0, 50, 10)

    if st.button("Predict Revenue"):

        input_data = np.array([[price, quantity, discount]])

        prediction = model.predict(input_data)

        st.success(f"Predicted Revenue: {prediction[0]:,.2f}")

# =========================
# REVENUE FORECAST
# =========================

elif menu == "Revenue Forecast":

    st.header("Revenue Forecast (ARIMA)")

    monthly = (
        df.set_index("order_date")
        .resample("M")["total_revenue"]
        .sum()
    )

    # Train ARIMA
    model = ARIMA(monthly, order=(1,1,1))

    model_fit = model.fit()

    forecast_period = st.slider("Forecast Period (Months)", 1, 12, 6)

    forecast = model_fit.forecast(steps=forecast_period)

    fig, ax = plt.subplots()

    monthly.plot(ax=ax, label="Historical")

    forecast.plot(ax=ax, label="Forecast")

    ax.legend()

    st.pyplot(fig)
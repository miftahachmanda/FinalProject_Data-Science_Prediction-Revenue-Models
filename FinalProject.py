import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Time Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Boosting
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# ==========================
# Load Data
# ==========================

df = pd.read_csv("amazon_sales_dataset.csv")

# Rapikan nama kolom
df.columns = df.columns.str.lower().str.strip()

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

    st.title("📊 E-Commerce Revenue Analytics Project")

    st.header("📌 Latar Belakang")

    st.write("""
Perkembangan e-commerce mendorong perusahaan tidak hanya fokus pada peningkatan jumlah transaksi, 
namun juga berupaya untuk mengoptimalkan revenue yang dihasilkan dari setiap transaksi.

Dengan banyaknya kategori produk, variasi harga, strategi diskon, serta perbedaan perilaku 
customer di setiap wilayah, perusahaan menghadapi tantangan dalam memahami faktor-faktor 
yang berkontribusi terhadap besarnya pendapatan.

Dataset yang digunakan merepresentasikan kondisi bisnis e-commerce melalui data transaksi 
berbasis waktu yang mencakup informasi produk, strategi diskon, wilayah pelanggan, serta 
rating dan ulasan.

Melalui pemanfaatan data analytics, perusahaan dapat mengubah data historis penjualan menjadi 
insight yang dapat mendukung pengambilan keputusan bisnis secara lebih akurat dan berbasis data.
""")

    st.header("🎯 Business Problem")

    st.write("""
Berdasarkan dataset transaksi yang tersedia, perusahaan ingin memahami:

1. Faktor-faktor apa saja yang paling berpengaruh terhadap total revenue pada setiap transaksi.
2. Bagaimana memprediksi revenue yang dihasilkan dari suatu transaksi berdasarkan karakteristik transaksi tersebut.
3. Bagaimana memprediksi total revenue di masa depan dengan menangkap pola trend dan seasonality penjualan.

Tanpa pemahaman tersebut, strategi harga dan diskon berpotensi belum optimal sehingga 
dapat menurunkan efektivitas peningkatan pendapatan.

Selain itu, keterbatasan dalam memprediksi revenue di masa depan menyulitkan perusahaan 
dalam melakukan perencanaan bisnis seperti:

- Penentuan target penjualan
- Strategi promosi
- Perencanaan persediaan
""")

    st.header("🎯 Tujuan Analisis")

    st.write("""
Analisis ini bertujuan untuk:

1️⃣ Memprediksi total revenue pada setiap transaksi menggunakan pendekatan Machine Learning.

2️⃣ Mengidentifikasi faktor-faktor utama yang paling berpengaruh terhadap revenue, seperti harga, 
jumlah produk terjual, dan diskon.

3️⃣ Melakukan forecasting revenue bulanan menggunakan metode time series untuk menangkap 
pola trend dan pola musiman pada penjualan.

4️⃣ Memberikan insight yang dapat membantu perusahaan dalam menyusun strategi bisnis yang 
lebih efektif dan berbasis data.
""")

    st.header("⚙️ Metodologi Analisis")

    st.write("""
Proses analisis dilakukan melalui beberapa tahapan utama:

1️⃣ Exploratory Data Analysis (EDA)  
Untuk memahami pola data, distribusi revenue, serta hubungan antar variabel.

2️⃣ Revenue Prediction (Machine Learning)  
Menggunakan model seperti:

- Decision Tree
- Random Forest
- XGBoost

Model ini digunakan untuk memprediksi revenue berdasarkan fitur transaksi seperti harga, jumlah produk, dan diskon.

3️⃣ Revenue Forecasting (Time Series)  
Untuk memprediksi revenue bulanan di masa depan menggunakan metode:

- ARIMA
- Exponential Smoothing
- XGBoost dengan lag features

4️⃣ Model Evaluation  
Model dievaluasi menggunakan metrik seperti:

- MAE
- RMSE
- R²
- MAPE
""")
    
# =========================
# DATA OVERVIEW
# =========================

if menu == "Data Overview":

    st.header("Dataset Overview")

    st.write("Jumlah Data:", df.shape)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    st.subheader("Kolom Dataset")
    st.write(list(df.columns))

    st.subheader("Statistical Summary")
    st.write(df.describe())


# =========================
# EDA
# =========================

elif menu == "EDA":

    st.header("Exploratory Data Analysis")

    df['order_date'] = pd.to_datetime(df['order_date'])

    df["total_revenue"] = df["price"] * df["quantity_sold"] * (1 - df["discount_percent"]/100)

    # =========================
    # Revenue Distribution
    # =========================

    st.subheader("Revenue Distribution")

    st.write(df["total_revenue"].describe())

    st.write(df["total_revenue"].agg(["mean", "median", "std", "min", "max", "skew"]))

    fig, ax = plt.subplots()

    sns.histplot(df["total_revenue"], bins=50, kde=True, ax=ax)

    ax.set_title("Distribution of Total Revenue")

    st.pyplot(fig)

    # =========================
    # Top Product Category
    # =========================

    st.subheader("Top Product Category")

    category_revenue = (
        df.groupby("product_category")["total_revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()

    category_revenue.plot(kind="bar", ax=ax)

    ax.set_title("Total Revenue by Product Category")

    st.pyplot(fig)

    st.write(category_revenue)

    st.info("Insight: Category Beauty menjadi kontributor revenue tertinggi.")

    # =========================
    # Quantity vs Revenue
    # =========================

    st.subheader("Quantity Sold vs Revenue")

    fig, ax = plt.subplots()

    sns.regplot(
        x="quantity_sold",
        y="total_revenue",
        data=df,
        scatter_kws={"alpha":0.5},
        line_kws={"color":"red"},
        ax=ax
    )

    ax.set_title("Quantity Sold vs Revenue")

    st.pyplot(fig)

    corr = df[["quantity_sold","total_revenue"]].corr()

    fig, ax = plt.subplots()

    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    st.pyplot(fig)

    st.info("Insight: Quantity sold memiliki korelasi positif terhadap revenue.")

    # =========================
    # Discount Analysis
    # =========================

    st.subheader("Discount Analysis")

    df["discount_flag"] = (df["discount_percent"] > 0).astype(int)

    fig, ax = plt.subplots()

    sns.boxplot(x="discount_flag", y="total_revenue", data=df, ax=ax)

    ax.set_xticklabels(["No Discount","Discount"])

    ax.set_title("Revenue: Discount vs Non-Discount")

    st.pyplot(fig)

    fig, ax = plt.subplots()

    sns.barplot(
        x="discount_flag",
        y="total_revenue",
        data=df,
        estimator="mean",
        ax=ax
    )

    ax.set_xticklabels(["No Discount","Discount"])

    ax.set_title("Average Revenue")

    st.pyplot(fig)

    st.info("Insight: Transaksi tanpa diskon memiliki average revenue lebih tinggi.")

    # =========================
    # Regional Analysis
    # =========================

    st.subheader("Revenue by Region")

    region_revenue = (
        df.groupby("customer_region")["total_revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()

    region_revenue.plot(kind="bar", ax=ax)

    ax.set_title("Total Revenue by Region")

    st.pyplot(fig)

    st.info("Insight: Middle East menjadi region dengan kontribusi revenue tertinggi.")

    # =========================
    # Payment Method
    # =========================

    st.subheader("Payment Method Preference")

    payment_region = (
        df.groupby(["customer_region","payment_method"])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10,6))

    payment_region.plot(kind="bar", stacked=True, ax=ax)

    st.pyplot(fig)

    # =========================
    # Rating vs Revenue
    # =========================

    st.subheader("Customer Rating vs Revenue")

    corr = df[["rating","total_revenue"]].corr()

    fig, ax = plt.subplots()

    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    st.pyplot(fig)

    st.info("Insight: Rating tidak memiliki korelasi signifikan terhadap revenue.")

    # =========================
    # Time Series Analysis
    # =========================

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
    # Seasonality
    # =========================

    st.subheader("Seasonality")

    monthly_avg = (
        df.groupby(df["order_date"].dt.month)["total_revenue"]
        .mean()
    )

    fig, ax = plt.subplots()

    monthly_avg.plot(kind="bar", ax=ax)

    st.pyplot(fig)

    peak_month = monthly_avg.idxmax()

    st.success(f"Peak Month Revenue: {peak_month}")

    # =========================
    # Revenue Fluctuation
    # =========================

    st.subheader("Revenue Fluctuation")

    fig, ax = plt.subplots()

    monthly_revenue.diff().plot(ax=ax)

    st.pyplot(fig)

    # =========================
    # Summary
    # =========================

    summary = {
        "Total Revenue": df["total_revenue"].sum(),
        "Average Revenue per Order": df["total_revenue"].mean(),
        "Total Orders": df["order_id"].nunique(),
        "Top Category": category_revenue.index[0],
        "Top Region": region_revenue.index[0],
        "Rating-Revenue Correlation": df[["rating","total_revenue"]].corr().iloc[0,1]
    }

    st.subheader("Business Summary")

    st.write(summary)


# =========================
# REVENUE PREDICTION
# =========================

elif menu == "Revenue Prediction":

    st.header("Revenue Prediction (Machine Learning)")

    required_cols = ["price", "quantity_sold", "discount_percent", "total_revenue"]

    if all(col in df.columns for col in required_cols):

        X = df[["price", "quantity_sold", "discount_percent"]]
        y = df["total_revenue"]

        model = RandomForestRegressor()
        model.fit(X, y)

        st.subheader("Input Transaction")

        price = st.number_input("Price", value=100)

        quantity = st.number_input("Quantity Sold", value=10)

        discount = st.slider("Discount (%)", 0, 50, 10)

        if st.button("Predict Revenue"):

            input_data = np.array([[price, quantity, discount]])

            prediction = model.predict(input_data)

            st.success(f"Predicted Revenue: {prediction[0]:,.2f}")

    else:

        st.error("Dataset tidak memiliki kolom yang dibutuhkan")


# =========================
# FORECASTING
# =========================

elif menu == "Revenue Forecast":

    st.header("Revenue Forecast (ARIMA)")

    if "total_revenue" in df.columns:

        if "month" in df.columns:

            monthly = df.groupby("month")["total_revenue"].sum()

        else:

            monthly = df["total_revenue"]

        model = ARIMA(monthly, order=(1,1,1))
        model_fit = model.fit()

        forecast_steps = st.slider("Forecast Period", 1, 12, 3)

        forecast = model_fit.forecast(steps=forecast_steps)

        fig, ax = plt.subplots()

        monthly.plot(ax=ax, label="Historical")
        forecast.plot(ax=ax, label="Forecast")

        ax.legend()

        st.pyplot(fig)

    else:

        st.error("Kolom total_revenue tidak ditemukan")
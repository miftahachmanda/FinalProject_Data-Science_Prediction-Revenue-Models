import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================
# Load Data
# ==========================

df = pd.read_csv("amazon_sales_dataset.csv")

df.columns = df.columns.str.lower().str.strip()

df["order_date"] = pd.to_datetime(df["order_date"])

# ==========================
# Feature Engineering
# ==========================

df["total_revenue"] = df["price"] * df["quantity_sold"] * (1 - df["discount_percent"]/100)

# ==========================
# Train Model
# ==========================

features = ["price","quantity_sold","discount_percent"]

X = df[features]
y = df["total_revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ==========================
# STREAMLIT DASHBOARD
# ==========================

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

# ==========================
# PROJECT OVERVIEW
# ==========================

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
    
# ==========================
# DATA OVERVIEW
# ==========================

elif menu == "Data Overview":

    st.header("Dataset Overview")

    st.write("Jumlah Data:", df.shape)

    st.subheader("Preview Data")

    st.dataframe(df.head())

    st.subheader("Statistical Summary")

    st.write(df.describe())

# ==========================
# EDA
# ==========================

elif menu == "EDA":

    st.header("Exploratory Data Analysis")

    # Revenue Distribution

    st.subheader("Revenue Distribution")

    fig, ax = plt.subplots()

    sns.histplot(df["total_revenue"], bins=50, kde=True, ax=ax)

    st.pyplot(fig)

    # Top Category

    st.subheader("Top Product Category")

    category_revenue = df.groupby("product_category")["total_revenue"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots()

    category_revenue.plot(kind="bar", ax=ax)

    st.pyplot(fig)

    st.write(category_revenue)

    # Quantity vs Revenue

    st.subheader("Quantity vs Revenue")

    fig, ax = plt.subplots()

    sns.regplot(
        x="quantity_sold",
        y="total_revenue",
        data=df,
        scatter_kws={"alpha":0.5},
        ax=ax
    )

    st.pyplot(fig)

    # Region Revenue

    st.subheader("Revenue by Region")

    region_revenue = df.groupby("customer_region")["total_revenue"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots()

    region_revenue.plot(kind="bar", ax=ax)

    st.pyplot(fig)

# ==========================
# REVENUE PREDICTION
# ==========================

elif menu == "Revenue Prediction":

    st.header("Revenue Prediction (Machine Learning)")

    st.write("Model yang digunakan: Random Forest Regressor")

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("R² Score", f"{r2:.3f}")

    st.divider()

    st.subheader("Simulasi Prediksi Revenue")

    price = st.number_input("Product Price", min_value=1.0, value=50.0)

    quantity = st.number_input("Quantity Sold", min_value=1, value=5)

    discount = st.slider("Discount (%)", 0, 50, 10)

    if st.button("Predict Revenue"):

        input_data = pd.DataFrame({
            "price":[price],
            "quantity_sold":[quantity],
            "discount_percent":[discount]
        })

        prediction = model.predict(input_data)

        st.success(f"Predicted Revenue: ${prediction[0]:,.2f}")

elif menu == "Revenue Forecast":

    st.header("Revenue Forecast (ARIMA)")

    monthly_revenue = (
        df.set_index("order_date")
        .resample("M")["total_revenue"]
        .sum()
    )

    model = joblib.load("best_forecasting_model.pkl")

    months = st.slider("Forecast Months", 1, 12, 3)

    forecast_log = model.forecast(steps=months)

    forecast = np.exp(forecast_log)

    forecast_index = pd.date_range(
        monthly_revenue.index[-1],
        periods=months + 1,
        freq="M"
    )[1:]

    forecast_series = pd.Series(
        forecast,
        index=forecast_index
    )

    # ==========================
    # Forecast Table
    # ==========================

    forecast_df = pd.DataFrame({
        "Month": forecast_series.index.strftime("%Y-%m"),
        "Forecast Revenue": forecast_series.values
    })

    st.subheader("Forecast Simulation Table")

    st.dataframe(forecast_df)

    # ==========================
    # Visualization
    # ==========================

    st.subheader("Revenue Forecast Visualization")

    fig, ax = plt.subplots()

    monthly_revenue[-12:].plot(
        label="Historical Revenue",
        marker="o",
        ax=ax
    )

    forecast_series.plot(
        label="Forecast Revenue",
        marker="o",
        ax=ax
    )

    ax.set_title("Revenue Forecast (ARIMA)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()

    st.pyplot(fig)
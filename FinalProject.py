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
    
    df['order_date'] = pd.to_datetime(df['order_date'])

    df["total_revenue"] = df["price"] * df["quantity_sold"] * (1 - df["discount_percent"]/100)

    # =========================
    # Split Dataset
    # =========================

    X = df.drop(columns=[
        'total_revenue',
        'order_id',
        'order_date',
        'product_id',
        'discounted_price',
        'revenue_check'
    ], errors='ignore')

    y = df['total_revenue']

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # =========================
    # Correlation EDA
    # =========================

    # st.subheader("Correlation Matrix")

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        pd.concat([x_train, y_train], axis=1)
        .select_dtypes(include=[np.number])
        .corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )



    # =========================
    # Feature Engineering
    # =========================

    #st.subheader("Feature Engineering")

    scaler = StandardScaler()

    num_cols = x_train.select_dtypes(include=[np.number]).columns

    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train[num_cols]),
        columns=num_cols,
        index=x_train.index
    )

    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test[num_cols]),
        columns=num_cols,
        index=x_test.index
    )

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    cat_cols = x_train.select_dtypes(include=['object']).columns

    x_train_encoded = pd.DataFrame(
        encoder.fit_transform(x_train[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=x_train.index
    )

    x_test_encoded = pd.DataFrame(
        encoder.transform(x_test[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=x_test.index
    )

    x_train_final = pd.concat([x_train_scaled, x_train_encoded], axis=1)
    x_test_final = pd.concat([x_test_scaled, x_test_encoded], axis=1)

    # =========================
    # Feature Selection
    # =========================

    #st.subheader("Feature Selection")

    selector_model = RandomForestRegressor(n_estimators=50, random_state=42)

    selector_model.fit(x_train_final, y_train)

    selector = SelectFromModel(selector_model, threshold="mean", prefit=True)

    x_train_sel = x_train_final.loc[:, selector.get_support()]
    x_test_sel = x_test_final.loc[:, selector.get_support()]


    # =========================
    # Model Training
    # =========================

    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    def evaluate_model(y_true, y_pred, name):

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return pd.DataFrame({
            "Model":[name],
            "MAE":[mae],
            "MSE":[mse],
            "RMSE":[rmse],
            "R2":[r2]
        })

    report = pd.DataFrame()

    for name, model in models.items():

        model.fit(x_train_sel, y_train)

        report = pd.concat([
            report,
            evaluate_model(
                y_test,
                model.predict(x_test_sel),
                name
            )
        ])



    # =========================
    # Feature Importance
    # =========================

    #st.subheader("Feature Importance (XGBoost)")

    importance = models["XGBoost"].feature_importances_

    df_importance = pd.DataFrame({
        "Feature": x_train_sel.columns,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots()

    sns.barplot(
        x="Importance",
        y="Feature",
        data=df_importance,
        ax=ax
    )



    # =========================
    # Hyperparameter Tuning
    # =========================

    # st.subheader("Hyperparameter Tuning (XGBoost)")

    param_grid = {
        'n_estimators':[100,200],
        'learning_rate':[0.01,0.05,0.1],
        'max_depth':[3,5,7],
        'subsample':[0.8,1.0]
    }

    grid_xgb = GridSearchCV(
        estimator=XGBRegressor(random_state=42),
        param_grid=param_grid,
        scoring="r2",
        cv=3,
        n_jobs=-1
    )

    grid_xgb.fit(x_train_sel, y_train)



    best_model = grid_xgb.best_estimator_

    y_pred = best_model.predict(x_test_sel)

    # =========================
    # Prediction vs Actual
    # =========================

    # st.subheader("Prediction vs Actual")

    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(y_test, y_pred, alpha=0.5)

    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())

    ax.plot([min_val,max_val],[min_val,max_val],'r--')



    # =========================
    # Residual Distribution
    # =========================

    # st.subheader("Residual Analysis")

    residuals = y_test - y_pred

    fig, ax = plt.subplots()

    sns.histplot(residuals, kde=True, ax=ax)

    ax.axvline(0, color="red", linestyle="--")




# =========================
# FORECASTING
# =========================

elif menu == "Revenue Forecast":

    st.header("Monthly Revenue Forecasting")

    # =====================
    # Data Preparation
    # =====================

    df['order_date'] = pd.to_datetime(df['order_date'])

    df['total_revenue'] = df['price'] * df['quantity_sold'] * (1 - df['discount_percent']/100)

    df = df.dropna()

    # =====================
    # Monthly Aggregation
    # =====================

    df_monthly = df.resample('M', on='order_date')['total_revenue'].sum().to_frame()
    df_monthly.columns = ['revenue']

    #st.subheader("Monthly Revenue")

    fig, ax = plt.subplots()

    ax.plot(df_monthly.index, df_monthly['revenue'])

    # st.pyplot(fig)

    # =====================
    # ADF Test
    # =====================

    result = adfuller(df_monthly['revenue'])

    #st.subheader("Stationarity Test (ADF)")

    #st.write("ADF Statistic:", result[0])
    #st.write("p-value:", result[1])

    # =====================
    # Train Test Split
    # =====================

    train = df_monthly.iloc[:-3]
    test = df_monthly.iloc[-3:]

    # =====================
    # Feature Engineering
    # =====================

    df_ml = df_monthly.copy()

    df_ml['lag_1'] = df_ml['revenue'].shift(1)
    df_ml['lag_2'] = df_ml['revenue'].shift(2)
    df_ml['rolling_mean_3'] = df_ml['revenue'].shift(1).rolling(3).mean()

    df_ml = df_ml.dropna()

    train_ml = df_ml.iloc[:-3]
    test_ml = df_ml.iloc[-3:]

    # =====================
    # MODEL 1: ARIMA
    # =====================

    model_arima = ARIMA(np.log(train['revenue']), order=(1,1,1))

    model_arima_fit = model_arima.fit()

    forecast_arima_log = model_arima_fit.forecast(steps=len(test))

    forecast_arima = np.exp(forecast_arima_log)

    # =====================
    # MODEL 2: ETS
    # =====================

    model_ets = ExponentialSmoothing(
        train['revenue'],
        trend='add',
        seasonal=None
    )

    model_ets_fit = model_ets.fit()

    forecast_ets = model_ets_fit.forecast(steps=len(test))

    # =====================
    # MODEL 3: XGBOOST
    # =====================

    model_xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )

    model_xgb.fit(
        train_ml[['lag_1','lag_2','rolling_mean_3']],
        train_ml['revenue']
    )

    forecast_xgb = model_xgb.predict(
        test_ml[['lag_1','lag_2','rolling_mean_3']]
    )

    # =====================
    # EVALUATION
    # =====================

    def evaluate_model(true, pred, name):

        mae = mean_absolute_error(true, pred)
        mape = mean_absolute_percentage_error(true, pred)*100

        return pd.DataFrame({
            "Model":[name],
            "MAE":[round(mae,2)],
            "MAPE":[round(mape,2)]
        })

    report = pd.concat([

        evaluate_model(test['revenue'], forecast_arima, "ARIMA"),
        evaluate_model(test['revenue'], forecast_ets, "ETS"),
        evaluate_model(test_ml['revenue'], forecast_xgb, "XGBoost")

    ])

   # st.subheader("Model Comparison")

    # st.dataframe(report)

    # =====================
    # Visualization
    # =====================

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(df_monthly.index[-12:], df_monthly['revenue'][-12:], label="Actual")

    ax.plot(test.index, forecast_arima, label="ARIMA")
    ax.plot(test.index, forecast_ets, label="ETS")
    ax.plot(test_ml.index, forecast_xgb, label="XGBoost")

    ax.legend()

    # st.pyplot(fig)

    # =====================
    # Best Model
    # =====================

    best_model = report.sort_values("MAPE").iloc[0]['Model']

    st.success(f"Best Model: {best_model}")

    # =====================
    # Future Forecast
    # =====================

    future_steps = st.slider("Forecast Months",1,12,3)

    if best_model == "ARIMA":

        final_model = ARIMA(np.log(df_monthly['revenue']), order=(1,1,1)).fit()

        future_log = final_model.forecast(steps=future_steps)

        future_forecast = np.exp(future_log)

    elif best_model == "ETS":

        final_model = ExponentialSmoothing(
            df_monthly['revenue'],
            trend='add'
        ).fit()

        future_forecast = final_model.forecast(steps=future_steps)

    st.subheader("Future Forecast")

    st.write(future_forecast)
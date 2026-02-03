
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Product Sales Dashboard", layout="wide")

st.title("📊 Product Sales Analysis (2023–2024)")

# Load dataset directly
@st.cache_data
def load_data():
    return pd.read_csv("product_sales_dataset_2023_2024.csv")

df = load_data()

# Dataset preview
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# Basic metrics
st.subheader("📌 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Total Products", df["Product"].nunique())

# Sales by Product
st.subheader("💰 Sales by Product")
product_sales = df.groupby("Product")["Sales"].sum().sort_values(ascending=False)

fig1, ax1 = plt.subplots()
product_sales.plot(kind="bar", ax=ax1)
ax1.set_xlabel("Product")
ax1.set_ylabel("Total Sales")
ax1.set_title("Total Sales per Product")
st.pyplot(fig1)

# Monthly Sales Trend
st.subheader("📈 Monthly Sales Trend")
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.to_period("M")

monthly_sales = df.groupby("Month")["Sales"].sum()

fig2, ax2 = plt.subplots()
monthly_sales.plot(kind="line", marker="o", ax=ax2)
ax2.set_xlabel("Month")
ax2.set_ylabel("Total Sales")
ax2.set_title("Monthly Sales Trend")
st.pyplot(fig2)

# Category-wise Sales
st.subheader("🗂 Category-wise Sales")
category_sales = df.groupby("Category")["Sales"].sum()

fig3, ax3 = plt.subplots()
category_sales.plot(kind="pie", autopct="%1.1f%%", ax=ax3)
ax3.set_ylabel("")
ax3.set_title("Sales Distribution by Category")
st.pyplot(fig3)

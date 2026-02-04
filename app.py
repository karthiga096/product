import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📊 Product Sales Analysis (2023–2024)",
    layout="wide"
)

st.title("📊 Product Sales Analysis (2023–2024)")
st.write("Automatic data loading with K-Means clustering")

# ---------------- LOAD DATA ----------------
FILE_NAME = "product_sales_dataset_final.csv"

if not os.path.exists(FILE_NAME):
    st.error("Dataset file not found ❌")
    st.info("Make sure 'product_sales_dataset_final.csv' is in the same folder as app.py")
    st.stop()

df = pd.read_csv(FILE_NAME)
st.success("Dataset loaded successfully ✅")

# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ---------------- FEATURES ----------------
features = ['Quantity', ' Unit_Price ', ' Revenue ', ' Profit ']

missing_cols = [c for c in features if c not in df.columns]
if missing_cols:
    st.error(f"Missing columns in dataset: {missing_cols}")
    st.stop()

X = df[features]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- K-MEANS ----------------
st.subheader("⚙️ K-Means Clustering")

k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ---------------- PCA ----------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# ---------------- VISUALIZATION ----------------
st.subheader("📈 Cluster Visualization (PCA)")

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='Cluster',
    data=df,
    palette='viridis',
    ax=ax
)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("K-Means Clustering Result")

st.pyplot(fig)

# ---------------- CLUSTER SUMMARY ----------------
st.subheader("📊 Cluster Summary")
st.dataframe(df.groupby('Cluster')[features].mean())

# ---------------- FOOTER ----------------
st.caption("Product Sales Analysis using Machine Learning (K-Means & PCA)")

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

# ---------------- LOAD DATA SILENTLY ----------------
FILE_NAME = "product_sales_dataset_final.csv"

if not os.path.exists(FILE_NAME):
    st.stop()   # stops app silently (NO message, NO error)

df = pd.read_csv(FILE_NAME)

# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ---------------- FEATURE SELECTION ----------------
features = ['Quantity', ' Unit_Price ', ' Revenue ', ' Profit ']
X = df[features]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- K-MEANS ----------------
k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ---------------- PCA ----------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# ---------------- VISUALIZATION ----------------
st.subheader("📈 K-Means Clustering (PCA)")

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
ax.set_title("Cluster Visualization")

st.pyplot(fig)

# ---------------- CLUSTER SUMMARY ----------------
st.subheader("📊 Cluster Summary")
st.dataframe(df.groupby('Cluster')[features].mean())

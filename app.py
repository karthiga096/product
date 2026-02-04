import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📊 Product Sales Analysis (2023–2024)",
    layout="wide"
)

st.title("📊 Product Sales Analysis (2023–2024)")
st.write("K-Means Clustering and Sales Insights")

# ---------------- FILE UPLOADER (NO ERROR) ----------------
st.subheader("📁 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload product_sales_dataset_2023_2024.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the CSV file to continue.")
    st.stop()   # ⛔ stops app safely (no error)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(uploaded_file)
st.success("Dataset loaded successfully ✅")

# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ---------------- FEATURE SELECTION ----------------
features = ['Quantity', ' Unit_Price ', ' Revenue ', ' Profit ']

# Check if all required columns exist
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns in dataset: {missing_cols}")
    st.stop()

X = df[features]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- K-MEANS CLUSTERING ----------------
st.subheader("⚙️ K-Means Clustering")

k = st.slider("Select number of clusters (k)", min_value=2, max_value=5, value=2)

kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(X_scaled)

st.success("Clustering completed ✅")

# ---------------- PCA ----------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# ---------------- VISUALIZATION ----------------
st.subheader("📈 Cluster Visualization using PCA")

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='Cluster',
    data=df,
    palette='viridis',
    ax=ax
)

ax.set_title("K-Means Clusters (PCA)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

st.pyplot(fig)

# ---------------- CLUSTER SUMMARY ----------------
st.subheader("📊 Cluster Summary (Mean Values)")
cluster_summary = df.groupby('Cluster')[features].mean()
st.dataframe(cluster_summary)

# ---------------- DOWNLOAD RESULT ----------------
st.subheader("⬇️ Download Clustered Dataset")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="product_sales_clustered_output.csv",
    mime="text/csv"
)

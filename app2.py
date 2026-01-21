import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation Predictor", layout="centered")

st.title("🧠 Customer Cluster Prediction (KMeans)")
st.write("Predict which customer cluster a user belongs to")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("marketing_campaign.csv")   # update if name differs
    return df

df = load_data()

# --------------------------------------------------
# Select features
# --------------------------------------------------
cat_features = ["Education", "Marital_Status"]
num_features = ["Income"]

df_model = df[cat_features + num_features].dropna()

# --------------------------------------------------
# Encoding categorical variables
# --------------------------------------------------
df_encoded = pd.get_dummies(df_model, columns=cat_features, drop_first=True)

# --------------------------------------------------
# Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# --------------------------------------------------
# Train KMeans
# --------------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
st.subheader("🔍 Enter Customer Details")

education = st.selectbox("Education", df["Education"].unique())
marital_status = st.selectbox("Marital Status", df["Marital_Status"].unique())
income = st.number_input("Income", min_value=0, step=1000)

# --------------------------------------------------
# Create input dataframe
# --------------------------------------------------
input_df = pd.DataFrame({
    "Education": [education],
    "Marital_Status": [marital_status],
    "Income": [income]
})

# One-hot encode input
input_encoded = pd.get_dummies(input_df, columns=cat_features, drop_first=True)

# Align columns with training data
input_encoded = input_encoded.reindex(columns=df_encoded.columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_encoded)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔮 Predict Cluster"):
    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"✅ This customer belongs to **Cluster {cluster}**")

    if cluster == 0:
        st.info("Cluster 0: Likely lower spending / conservative customers")
    else:
        st.info("Cluster 1: Likely higher spending / responsive customers")

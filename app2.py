import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
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
    df = pd.read_csv("C:\Users\neeth\Downloads\marketing_campaign.csv")   # update if name differs
    return df

df = load_data()

df_filled = df.fillna(df.mean(numeric_only=True))
# --------------------------------------------------
# Outlier capping 
# --------------------------------------------------
 
def outlier_capping(df_filled, column): 
    Q1 = df_filled[column].quantile(0.25) 
    Q3 = df_filled[column].quantile(0.75) 
    IQR = Q3-Q1 
    l_e = Q1-1.5*IQR 
    u_e = Q3+1.5*IQR 
    
    df_filled[column] = df_filled[column].apply(lambda x: l_e if x<l_e else u_e if x>u_e else x) 

for col in df_filled.select_dtypes(['int', 'float']): 
    outlier_capping(df_filled, col)
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
# Feature engineering
# --------------------------------------------------
current_year = datetime.now().year
df_filled['Age'] = current_year - df_filled['Year_Birth']

df_filled.drop(columns=['Year_Birth'], inplace=True)

df_filled['Children'] = df_filled['Kidhome'] + df_filled['Teenhome']

df_filled['TotalSpend'] = (df_filled['MntWines'] + df_filled['MntFruits'] + df_filled['MntMeatProducts'] + df_filled['MntFishProducts'] + df_filled['MntSweetProducts'] + 
                    df_filled['MntGoldProds'])

df_filled['TotalPurchases'] = (df_filled['NumWebPurchases'] + df_filled['NumCatalogPurchases'] + df_filled['NumStorePurchases'])

# --------------------------------------------------
# Scaling
# --------------------------------------------------

num_cols = df_filled.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_filled.select_dtypes(include=['object']).columns.tolist()

scaler_standard = StandardScaler()
df_standard = df_filled.copy(num_cols)
df_standard[num_cols] = scaler_standard.fit_transform(df_standard[num_cols])

# --------------------------------------------------
# Train KMeans
# --------------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(df_standard[num_cols])

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
input_scaled = scaler_standard.transform(input_encoded)

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

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Customer AI Dashboard", layout="wide")

# -----------------------------
# CUSTOM UI
# -----------------------------
st.markdown("""
    <style>
    .card {
        background: #1c1f26;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.title("Customer AI Segmentation Dashboard")
st.caption("AI-powered customer segmentation & insights")

# -----------------------------
# UPLOAD DATA
# -----------------------------
uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # FEATURE SELECTION
    # -----------------------------
    features = ['Annual Income', 'Spending Score']

    if not all(col in df.columns for col in features):
        st.error("Dataset must contain 'Annual Income' and 'Spending Score'")
        st.stop()

    X = df[features]

    # -----------------------------
    # SCALING (IMPORTANT FIX)
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # SIDEBAR
    # -----------------------------
    k = st.slider("Select Number of Clusters", 2, 10, 5)

    # -----------------------------
    # MODEL
    # -----------------------------
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(X_scaled)

    # -----------------------------
    # KPI CARDS
    # -----------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"<div class='card'>👥<br><b>{len(df)}</b><br>Total Customers</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'>💰<br><b>{df['Annual Income'].mean():.2f}</b><br>Avg Income</div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'>🛍️<br><b>{df['Spending Score'].mean():.2f}</b><br>Avg Spending</div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='card'>📊<br><b>{k}</b><br>Clusters</div>", unsafe_allow_html=True)

    # -----------------------------
    # LAYOUT
    # -----------------------------
    left, right = st.columns([1, 2])

    # -----------------------------
    # INSIGHTS
    # -----------------------------
    with left:
        st.subheader("🧠 Cluster Insights")

        numeric_summary = df.groupby("Cluster")[features].mean()
        st.dataframe(numeric_summary)

        st.download_button(
            "📥 Download Segmented Data",
            df.to_csv(index=False),
            file_name="segmented_customers.csv",
            mime="text/csv"
        )

    # -----------------------------
    # VISUALIZATION (PRO LEVEL)
    # -----------------------------
    with right:
        st.subheader("📈 Customer Segmentation Map")

        fig = px.scatter(
            df,
            x="Annual Income",
            y="Spending Score",
            color=df["Cluster"].astype(str),
            title="Customer Clusters",
            opacity=0.7
        )

        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # CLUSTER MEANING
    # -----------------------------
    st.subheader("💡 Business Interpretation")

    st.info("""
    - 🔴 High Income + High Spending → VIP Customers  
    - 🟡 Low Income + High Spending → Impulsive Buyers  
    - 🔵 High Income + Low Spending → Target Marketing Group  
    - 🟢 Low Income + Low Spending → Budget Customers  
    """)

else:
    st.info("Upload a dataset to start analysis 🚀")

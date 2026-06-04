# ============================================================
# Streamlit Dashboard — Parcl Buyer Segmentation
# Author: aryajaitapkar21-ship-it
# Mentor: Mr. Sandeep Sir (Unified Mentor)
# Date: May 2026
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Parcl Buyer Segmentation",
    page_icon="🏠",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    .block-container { padding-top: 1rem; }
    h1 { color: #6366f1; }
    h2, h3 { color: #a5b4fc; }
    .metric-container { background: #111827; border-radius: 10px; padding: 1rem; }
    .stMetric { background: #111827; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.title("🏠 Parcl — Buyer Segmentation Intelligence")
st.markdown("**Machine Learning Based Buyer Segmentation | Unified Mentor × Parcl Co. Limited | May 2026**")
st.markdown("---")

# ── Load & Process Data ──────────────────────────────────────
@st.cache_data
def load_and_cluster():
    clients = pd.read_csv('clients.csv')
    props   = pd.read_csv('properties.csv')

    # Clean
    clients['dob'] = pd.to_datetime(clients['date_of_birth'], errors='coerce')
    clients['age'] = (pd.Timestamp('2026-05-01') - clients['dob']).dt.days // 365
    clients['age'] = clients['age'].fillna(clients['age'].median())
    props['price'] = props['sale_price'].str.replace(r'[\$,]', '', regex=True).astype(float)

    # Feature engineering
    props_sold = props.dropna(subset=['client_ref'])
    client_props = props_sold.groupby('client_ref').agg(
        num_transactions=('listing_id','count'),
        avg_price=('price','mean'),
        total_spent=('price','sum'),
        avg_area=('floor_area_sqft','mean')
    ).reset_index()
    client_props.columns = ['client_id','num_transactions','avg_price','total_spent','avg_area']

    df = clients.merge(client_props, on='client_id', how='left')
    df['num_transactions'] = df['num_transactions'].fillna(0)
    df['avg_price']        = df['avg_price'].fillna(df['avg_price'].median())
    df['total_spent']      = df['total_spent'].fillna(0)
    df['avg_area']         = df['avg_area'].fillna(df['avg_area'].median())

    # Encode
    le = LabelEncoder()
    df['client_type_enc'] = le.fit_transform(df['client_type'])
    df['acq_purpose_enc'] = le.fit_transform(df['acquisition_purpose'])
    df['loan_enc']        = le.fit_transform(df['loan_applied'])
    df['referral_enc']    = le.fit_transform(df['referral_channel'])

    # Scale & Cluster
    features = ['age','satisfaction_score','num_transactions','avg_price',
                'total_spent','avg_area','client_type_enc','acq_purpose_enc',
                'loan_enc','referral_enc']
    X = df[features].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_x'] = X_pca[:, 0]
    df['pca_y'] = X_pca[:, 1]

    return df, props

df, props = load_and_cluster()

CLUSTER_NAMES  = {0:"🏦 Serial Investors", 1:"💎 Luxury Buyers", 2:"🏠 Loan-Reliant", 3:"🏘️ Mass Market"}
CLUSTER_COLORS = {0:"#6366f1", 1:"#10b981", 2:"#f59e0b", 3:"#ef4444"}

# ── Sidebar Filters ──────────────────────────────────────────
st.sidebar.title("🔍 Filters")
selected_cluster = st.sidebar.selectbox("Cluster", ["All"] + list(CLUSTER_NAMES.values()))
selected_country = st.sidebar.selectbox("Country", ["All"] + sorted(df['country'].unique().tolist()))
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", ["All"] + sorted(df['acquisition_purpose'].unique().tolist()))
selected_type    = st.sidebar.selectbox("Client Type", ["All"] + sorted(df['client_type'].unique().tolist()))

# Apply filters
filtered = df.copy()
if selected_cluster != "All":
    cnum = [k for k,v in CLUSTER_NAMES.items() if v == selected_cluster][0]
    filtered = filtered[filtered['cluster'] == cnum]
if selected_country != "All":
    filtered = filtered[filtered['country'] == selected_country]
if selected_purpose != "All":
    filtered = filtered[filtered['acquisition_purpose'] == selected_purpose]
if selected_type != "All":
    filtered = filtered[filtered['client_type'] == selected_type]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing: {len(filtered)} clients**")

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Investor Behavior", "🗺️ Geography", "🔬 ML Methodology"])

# ════════ TAB 1 — OVERVIEW ════════
with tab1:
    st.header("Buyer Segment Overview")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    cluster_data = [
        (0, 94,   57.0, 339526,  2096925, "6.2 transactions"),
        (1, 762,  52.5, 413532,  1443678, "$413K avg price"),
        (2, 102,  50.0, 345481,  1240326, "42% loan rate"),
        (3, 1042, 52.8, 299342,  1052826, "Largest group"),
    ]
    for col, (ci, count, age, price, spend, note) in zip([col1,col2,col3,col4], cluster_data):
        with col:
            st.metric(
                label=CLUSTER_NAMES[ci],
                value=f"{count} clients",
                delta=note
            )

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Cluster Distribution")
        dist = filtered['cluster'].value_counts().reset_index()
        dist.columns = ['cluster','count']
        dist['name'] = dist['cluster'].map(CLUSTER_NAMES)
        st.bar_chart(dist.set_index('name')['count'])

    with col_b:
        st.subheader("Avg Price per Cluster ($)")
        price_data = filtered.groupby('cluster')['avg_price'].mean().reset_index()
        price_data['name'] = price_data['cluster'].map(CLUSTER_NAMES)
        st.bar_chart(price_data.set_index('name')['avg_price'])

    st.markdown("---")
    st.subheader("PCA 2D Cluster Visualization")

    import random
    random.seed(42)
    pca_sample = filtered.sample(min(300, len(filtered)), random_state=42)

    scatter_data = pd.DataFrame({
        'PCA Component 1': pca_sample['pca_x'],
        'PCA Component 2': pca_sample['pca_y'],
        'Cluster': pca_sample['cluster'].map(CLUSTER_NAMES)
    })
    st.scatter_chart(scatter_data, x='PCA Component 1', y='PCA Component 2', color='Cluster')

# ════════ TAB 2 — BEHAVIOR ════════
with tab2:
    st.header("Investor Behavior Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Loan Uptake by Cluster (%)")
        loan_data = filtered.groupby('cluster')['loan_enc'].mean().reset_index()
        loan_data['Loan %'] = (loan_data['loan_enc'] * 100).round(1)
        loan_data['Cluster'] = loan_data['cluster'].map(CLUSTER_NAMES)
        st.bar_chart(loan_data.set_index('Cluster')['Loan %'])

    with col2:
        st.subheader("Avg Satisfaction Score")
        sat_data = filtered.groupby('cluster')['satisfaction_score'].mean().reset_index()
        sat_data['Cluster'] = sat_data['cluster'].map(CLUSTER_NAMES)
        st.bar_chart(sat_data.set_index('Cluster')['satisfaction_score'])

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Acquisition Purpose Distribution")
        acq = filtered.groupby(['acquisition_purpose','cluster']).size().reset_index(name='count')
        acq['Cluster'] = acq['cluster'].map(CLUSTER_NAMES)
        acq_pivot = acq.pivot(index='Cluster', columns='acquisition_purpose', values='count').fillna(0)
        st.bar_chart(acq_pivot)

    with col4:
        st.subheader("Referral Channel Distribution")
        ref = filtered.groupby(['referral_channel','cluster']).size().reset_index(name='count')
        ref['Cluster'] = ref['cluster'].map(CLUSTER_NAMES)
        ref_pivot = ref.pivot(index='Cluster', columns='referral_channel', values='count').fillna(0)
        st.bar_chart(ref_pivot)

    st.markdown("---")
    st.subheader("Detailed Cluster Statistics")
    summary = filtered.groupby('cluster').agg(
        Count=('client_id','count'),
        Avg_Age=('age','mean'),
        Avg_Satisfaction=('satisfaction_score','mean'),
        Avg_Price=('avg_price','mean'),
        Avg_Total_Spend=('total_spent','mean'),
        Avg_Transactions=('num_transactions','mean'),
        Loan_Pct=('loan_enc','mean'),
    ).round(2).reset_index()
    summary['Cluster'] = summary['cluster'].map(CLUSTER_NAMES)
    summary = summary.drop('cluster', axis=1).set_index('Cluster')
    st.dataframe(summary, use_container_width=True)

# ════════ TAB 3 — GEOGRAPHY ════════
with tab3:
    st.header("Geographic Buyer Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Countries by Buyer Count")
        country_data = filtered['country'].value_counts().head(10).reset_index()
        country_data.columns = ['Country','Count']
        st.bar_chart(country_data.set_index('Country'))

    with col2:
        st.subheader("Top Regions by Buyer Count")
        region_data = filtered['region'].value_counts().head(10).reset_index()
        region_data.columns = ['Region','Count']
        st.bar_chart(region_data.set_index('Region'))

    st.markdown("---")
    st.subheader("Cluster Distribution by Country (Top 8)")
    top_countries = filtered['country'].value_counts().head(8).index.tolist()
    country_cluster = filtered[filtered['country'].isin(top_countries)]
    cc = country_cluster.groupby(['country','cluster']).size().reset_index(name='count')
    cc['Cluster'] = cc['cluster'].map(CLUSTER_NAMES)
    cc_pivot = cc.pivot(index='country', columns='Cluster', values='count').fillna(0)
    st.bar_chart(cc_pivot)

# ════════ TAB 4 — ML METHODOLOGY ════════
with tab4:
    st.header("ML Methodology")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Pipeline Steps")
        steps = [
            "1️⃣ Data Cleaning — Age parsing, price formatting",
            "2️⃣ Feature Engineering — Cross-dataset aggregation",
            "3️⃣ Label Encoding — 4 categorical features",
            "4️⃣ StandardScaler — Normalize all features",
            "5️⃣ K-Means Clustering — k=2 to k=8 tested",
            "6️⃣ Elbow + Silhouette — k=4 selected (0.1839)",
            "7️⃣ PCA 2D — Cluster visualization",
        ]
        for s in steps:
            st.markdown(s)

        st.markdown("---")
        st.subheader("🎯 Feature Set (10 Variables)")
        features_df = pd.DataFrame({
            'Feature': ['age','satisfaction_score','num_transactions','avg_price',
                       'total_spent','avg_area','client_type','acq_purpose','loan_applied','referral_channel'],
            'Type': ['Numeric','Numeric','Numeric','Numeric','Numeric','Numeric',
                    'Encoded','Encoded','Encoded','Encoded'],
        })
        st.dataframe(features_df, use_container_width=True)

    with col2:
        st.subheader("📈 Elbow Method")
        elbow_df = pd.DataFrame({
            'k': [2,3,4,5,6,7,8],
            'Inertia': [16719,14882,13545,12971,11514,10898,10303]
        }).set_index('k')
        st.line_chart(elbow_df)

        st.subheader("📊 Silhouette Scores")
        sil_df = pd.DataFrame({
            'k': [2,3,4,5,6,7,8],
            'Silhouette': [0.1601,0.1781,0.1839,0.1270,0.1540,0.1488,0.1576]
        }).set_index('k')
        st.line_chart(sil_df)

        st.success("✅ Optimal k=4 selected (Silhouette: 0.1839)")

    st.markdown("---")
    st.subheader("👥 Discovered Segments")
    segments_df = pd.DataFrame({
        'Segment': ['🏦 Serial Investors','💎 Luxury Buyers','🏠 Loan-Reliant','🏘️ Mass Market'],
        'Count': [94, 762, 102, 1042],
        'Avg Age': [57.0, 52.5, 50.0, 52.8],
        'Avg Price ($)': [339526, 413532, 345481, 299342],
        'Total Spend ($)': [2096925, 1443678, 1240326, 1052826],
        'Avg Transactions': [6.23, 3.51, 3.61, 3.53],
        'Loan %': ['36%','36%','42%','37%'],
        'Satisfaction': [3.59, 3.05, 3.05, 2.96],
    })
    st.dataframe(segments_df, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("**© 2026 Parcl Co. Limited × Unified Mentor | Mentor: Mr. Sandeep Sir | Project by aryajaitapkar21-ship-it**")

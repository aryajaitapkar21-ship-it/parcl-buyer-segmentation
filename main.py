# ============================================================
# Machine Learning Based Buyer Segmentation
# Parcl Co. Limited x Unified Mentor
# Author: aryajaitapkar21-ship-it
# Mentor: Mr. Sandeep Sir (Unified Mentor)
# Date: May 2026
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ── STEP 1: Load Data ────────────────────────────────────────
print("Loading data...")
clients = pd.read_csv('clients.csv')
props   = pd.read_csv('properties.csv')
print(f"Clients: {clients.shape} | Properties: {props.shape}")

# ── STEP 2: Data Cleaning ────────────────────────────────────
print("\nCleaning data...")

# Parse age from date_of_birth
clients['dob'] = pd.to_datetime(clients['date_of_birth'], errors='coerce')
clients['age'] = (pd.Timestamp('2026-05-01') - clients['dob']).dt.days // 365
clients['age'] = clients['age'].fillna(clients['age'].median())

# Clean price from string format e.g. "$300,000.00" -> 300000.0
props['price'] = props['sale_price'].str.replace(r'[\$,]', '', regex=True).astype(float)

print(f"Age range: {clients['age'].min():.0f} - {clients['age'].max():.0f} years")
print(f"Price range: ${props['price'].min():,.0f} - ${props['price'].max():,.0f}")

# ── STEP 3: Feature Engineering ─────────────────────────────
print("\nEngineering features...")

# Keep only sold properties (those with a client reference)
props_sold = props.dropna(subset=['client_ref'])
print(f"Sold properties: {len(props_sold)} / {len(props)}")

# Aggregate per-client property statistics
client_props = props_sold.groupby('client_ref').agg(
    num_transactions = ('listing_id', 'count'),
    avg_price        = ('price', 'mean'),
    total_spent      = ('price', 'sum'),
    avg_area         = ('floor_area_sqft', 'mean')
).reset_index()
client_props.columns = ['client_id', 'num_transactions', 'avg_price', 'total_spent', 'avg_area']

# Merge with client data
df = clients.merge(client_props, on='client_id', how='left')

# Fill missing values for clients with no transactions
df['num_transactions'] = df['num_transactions'].fillna(0)
df['avg_price']        = df['avg_price'].fillna(df['avg_price'].median())
df['total_spent']      = df['total_spent'].fillna(0)
df['avg_area']         = df['avg_area'].fillna(df['avg_area'].median())

print(f"Merged dataset shape: {df.shape}")

# ── STEP 4: Encoding ─────────────────────────────────────────
print("\nEncoding categorical features...")

le = LabelEncoder()
df['client_type_enc']  = le.fit_transform(df['client_type'])        # Individual=1, Company=0
df['acq_purpose_enc']  = le.fit_transform(df['acquisition_purpose']) # Home=0, Investment=1
df['loan_enc']         = le.fit_transform(df['loan_applied'])        # No=0, Yes=1
df['referral_enc']     = le.fit_transform(df['referral_channel'])    # Agency=0, Client=1, Website=2

# ── STEP 5: Feature Selection & Scaling ─────────────────────
features = [
    'age', 'satisfaction_score', 'num_transactions',
    'avg_price', 'total_spent', 'avg_area',
    'client_type_enc', 'acq_purpose_enc', 'loan_enc', 'referral_enc'
]

X = df[features].values
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Feature matrix shape: {X_scaled.shape}")

# ── STEP 6: Optimal K Selection ──────────────────────────────
print("\nFinding optimal number of clusters (k=2 to 8)...")
print(f"{'k':<5} {'Inertia':<12} {'Silhouette':<12}")
print("-" * 30)

inertias, silhouettes = [], []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)
    marker = " <-- OPTIMAL" if k == 4 else ""
    print(f"{k:<5} {km.inertia_:<12.1f} {sil:<12.4f}{marker}")

# ── STEP 7: Final K-Means Model (k=4) ────────────────────────
print("\nTraining final model with k=4...")
km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = km_final.fit_predict(X_scaled)

final_sil = silhouette_score(X_scaled, df['cluster'])
print(f"Final Silhouette Score: {final_sil:.4f}")
print(f"Final Inertia: {km_final.inertia_:.2f}")

# ── STEP 8: PCA for Visualization ────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca_x'] = X_pca[:, 0]
df['pca_y'] = X_pca[:, 1]
print(f"\nPCA Variance Explained: {pca.explained_variance_ratio_[0]:.2%} + {pca.explained_variance_ratio_[1]:.2%} = {sum(pca.explained_variance_ratio_):.2%}")

# ── STEP 9: Cluster Analysis ──────────────────────────────────
CLUSTER_NAMES = {
    0: "Serial Investors",
    1: "Luxury Buyers",
    2: "Loan-Reliant Buyers",
    3: "Mass Market Buyers"
}

print("\n" + "="*60)
print("CLUSTER SUMMARY RESULTS")
print("="*60)

for c in range(4):
    sub = df[df['cluster'] == c]
    print(f"\nCluster {c} — {CLUSTER_NAMES[c]}")
    print(f"  Count          : {len(sub)} clients ({len(sub)/len(df)*100:.1f}%)")
    print(f"  Avg Age        : {sub['age'].mean():.1f} years")
    print(f"  Avg Satisfaction: {sub['satisfaction_score'].mean():.2f} / 5")
    print(f"  Avg Unit Price : ${sub['avg_price'].mean():,.0f}")
    print(f"  Avg Total Spend: ${sub['total_spent'].mean():,.0f}")
    print(f"  Avg Transactions: {sub['num_transactions'].mean():.2f}")
    print(f"  Loan Uptake    : {sub['loan_enc'].mean()*100:.0f}%")
    print(f"  Investment %   : {sub['acq_purpose_enc'].mean()*100:.0f}%")

# ── STEP 10: Save Results ─────────────────────────────────────
output_file = 'segmentation_results.csv'
df[['client_id', 'cluster', 'age', 'satisfaction_score',
    'avg_price', 'total_spent', 'num_transactions',
    'client_type', 'acquisition_purpose', 'loan_applied',
    'referral_channel', 'country', 'region', 'pca_x', 'pca_y']].to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
print("\nDone! ✓")

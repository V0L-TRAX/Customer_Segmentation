"""
Task 04 - Customer Segmentation Analysis
========================================
This script performs customer segmentation using K-Means clustering
on the Mall Customers dataset.

Features used: Annual Income + Spending Score
Algorithm   : K-Means (with Elbow Method + Silhouette Score)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (avoids Windows display issues)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# --------------------------------------------------
# Setup paths (works on Windows + Linux/Mac)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"

# Create folders if they don't exist
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# Set style (safe fallback)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        plt.style.use('ggplot')

sns.set_palette("viridis")

# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------
print("=" * 60)
print("CUSTOMER SEGMENTATION ANALYSIS")
print("=" * 60)

data_path = DATA_DIR / "Mall_Customers.csv"
df = pd.read_csv(data_path)

print("\n[1] Dataset Loaded Successfully")
print(f"    Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# --------------------------------------------------
# 2. Basic EDA
# --------------------------------------------------
print("\n[2] Checking for missing values...")
print(df.isnull().sum())

print("\nGender Distribution:")
print(df['Gender'].value_counts())

# --------------------------------------------------
# 3. Feature Selection & Scaling
# --------------------------------------------------
print("\n[3] Selecting features: Annual Income + Spending Score")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("    Features scaled using StandardScaler")

# --------------------------------------------------
# 4. Find Optimal Number of Clusters
# --------------------------------------------------
print("\n[4] Finding optimal number of clusters (Elbow + Silhouette)...")

wcss = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"    k={k} | WCSS={kmeans.inertia_:.2f} | Silhouette={score:.4f}")

# Plot Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, wcss, marker='o', linewidth=2, markersize=8, color='#2E86AB')
axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('WCSS (Inertia)')
axes[0].axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Optimal k=5')
axes[0].legend()

axes[1].plot(K_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='#28A745')
axes[1].set_title('Silhouette Score', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Optimal k=5')
axes[1].legend()

plt.tight_layout()
elbow_path = IMAGES_DIR / "elbow_silhouette.png"
plt.savefig(str(elbow_path), dpi=150, bbox_inches='tight')
print(f"\n    Saved: {elbow_path}")
plt.close()

# --------------------------------------------------
# 5. Train Final K-Means Model (k=5)
# --------------------------------------------------
print("\n[5] Training final K-Means model with k=5...")

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

print(f"    Model trained. Cluster distribution:")
print(df['Cluster'].value_counts().sort_index())

# --------------------------------------------------
# 6. Visualize Clusters
# --------------------------------------------------
print("\n[6] Creating cluster visualization...")

# Inverse transform centroids for plotting in original scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(11, 8))
colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#9B5DE5']

for i in range(optimal_k):
    plt.scatter(
        X[clusters == i, 0],
        X[clusters == i, 1],
        s=80,
        c=colors[i],
        label=f'Cluster {i}',
        alpha=0.75,
        edgecolors='white',
        linewidth=0.5
    )

# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=350,
    c='black',
    marker='X',
    label='Centroids',
    edgecolors='white',
    linewidth=1.5
)

plt.title('Customer Segments (K-Means Clustering)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend(title='Segments', loc='upper right', frameon=True)
plt.grid(True, alpha=0.3)

cluster_path = IMAGES_DIR / "customer_clusters.png"
plt.savefig(str(cluster_path), dpi=150, bbox_inches='tight')
print(f"    Saved: {cluster_path}")
plt.close()

# --------------------------------------------------
# 7. Cluster Profiling & Business Interpretation
# --------------------------------------------------
print("\n[7] Cluster Profiling & Business Interpretation")
print("-" * 60)

cluster_summary = df.groupby('Cluster').agg({
    'Age': ['mean', 'min', 'max'],
    'Annual Income (k$)': ['mean', 'min', 'max'],
    'Spending Score (1-100)': ['mean', 'min', 'max'],
    'CustomerID': 'count'
}).round(1)

cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
cluster_summary = cluster_summary.rename(columns={'CustomerID_count': 'Count'})
print(cluster_summary)

print("\nGender distribution per cluster:")
print(pd.crosstab(df['Cluster'], df['Gender'], normalize='index').round(2) * 100)

print("\n" + "=" * 60)
print("SEGMENT INTERPRETATION (based on Income vs Spending)")
print("=" * 60)

for i in range(optimal_k):
    avg_income = df[df['Cluster'] == i]['Annual Income (k$)'].mean()
    avg_spend = df[df['Cluster'] == i]['Spending Score (1-100)'].mean()
    count = (df['Cluster'] == i).sum()

    if avg_income > 70 and avg_spend > 60:
        name = "Premium / VIP Customers"
        action = "Loyalty programs, exclusive offers, premium products"
    elif avg_income < 40 and avg_spend > 60:
        name = "High Spenders (Low Income) - Impulse Buyers"
        action = "Value deals, discounts, limited-time offers"
    elif avg_income > 70 and avg_spend < 40:
        name = "Potential / Target Customers (High Income, Low Spend)"
        action = "Upselling campaigns, personalized recommendations"
    elif avg_income < 40 and avg_spend < 40:
        name = "Budget / Low-Value Customers"
        action = "Cost-effective retention, basic promotions"
    else:
        name = "Average / Standard Customers"
        action = "Standard marketing campaigns"

    print(f"\nCluster {i}: {name}")
    print(f"  • Size           : {count} customers ({count/len(df)*100:.1f}%)")
    print(f"  • Avg Income     : ${avg_income:.1f}k")
    print(f"  • Avg Spend Score: {avg_spend:.1f}")
    print(f"  • Recommended Action: {action}")

# --------------------------------------------------
# 8. Additional Visualization - Boxplots
# --------------------------------------------------
print("\n[8] Creating additional charts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(x='Cluster', y='Spending Score (1-100)', data=df, ax=axes[0], palette=colors)
axes[0].set_title('Spending Score by Cluster', fontweight='bold')

sns.boxplot(x='Cluster', y='Annual Income (k$)', data=df, ax=axes[1], palette=colors)
axes[1].set_title('Annual Income by Cluster', fontweight='bold')

plt.tight_layout()
box_path = IMAGES_DIR / "cluster_boxplots.png"
plt.savefig(str(box_path), dpi=150, bbox_inches='tight')
print(f"    Saved: {box_path}")
plt.close()

# Save clustered data
output_csv = DATA_DIR / "customers_with_clusters.csv"
df.to_csv(output_csv, index=False)
print(f"\n    Clustered data saved to: {output_csv}")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nGenerated files:")
print(f"  • {IMAGES_DIR / 'elbow_silhouette.png'}")
print(f"  • {IMAGES_DIR / 'customer_clusters.png'}")
print(f"  • {IMAGES_DIR / 'cluster_boxplots.png'}")
print(f"  • {output_csv}")
print("\nYou can now use the segmented data for targeted marketing.")

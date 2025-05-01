import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset (replace with your file path if using a custom dataset)
df = pd.read_excel('file-DWGmhvuqzqe89RnCW7F32d')  # Example: uploaded Excel file

# Drop non-numeric columns and rows with missing values
df = df.select_dtypes(include=[np.number]).dropna()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Determine optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# Apply K-Means with chosen number of clusters (e.g., k=3)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to original data
df['Cluster'] = clusters

# Use PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Visualize clusters.
plt.figure(figsize=(8, 6))
for cluster in range(k):
    plt.scatter(df[df['Cluster'] == cluster]['PCA1'], 
                df[df['Cluster'] == cluster]['PCA2'], 
                label=f'Cluster {cluster}')
plt.title("K-Means Clustering Visualization (PCA-reduced)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.grid(True)
plt.show()

# Wholesale Customers Clustering Analysis

**Data Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/292/wholesale+customers)  
**GitHub Repository:** https://github.com/GHeart01/Wholesale_Customer_Clustering  
**YouTube Demo:** https://youtu.be/_jKNcg4YtCs

## Table of Contents
- [Description](#description)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [KMeans Clustering Model](#kmeans-clustering-model)
- [Hierarchical Clustering Model](#hierarchical-clustering-model)
- [Model Optimization](#model-optimization)
- [Results and Analysis](#results-and-analysis)
- [Conclusion](#conclusion)
- [Citation](#citation)

## Description

This dataset from the UC Irvine Machine Learning Repository contains information about wholesale customers' annual spending across different product categories. Since there is no target variable, the goal is to identify customer segments through unsupervised clustering based on purchasing behavior patterns.

The analysis employs both KMeans and Hierarchical clustering algorithms, with model quality assessed using silhouette scores.

![Silhouette Score Comparison](SilhouetteScore.png)

## Exploratory Data Analysis

### Dataset Features

| Feature            | Description                                                          |     
|--------------------|----------------------------------------------------------------------|
| Channel            | Sales channel - Horeca (Hotel/Restaurant/Cafe) or Retail           | 
| Region             | Geographic region - Lisbon, Oporto, or Other                       |
| Fresh              | Annual spending (monetary units) on fresh products                  | 
| Milk               | Annual spending (monetary units) on milk products                   | 
| Grocery            | Annual spending (monetary units) on grocery products                | 
| Frozen             | Annual spending (monetary units) on frozen products                 | 
| Detergents_Paper   | Annual spending (monetary units) on detergents and paper products   | 
| Delicassen         | Annual spending (monetary units) on delicatessen products           | 

### Correlation Analysis

The categorical columns (Channel and Region) are excluded from correlation analysis as they don't provide meaningful statistical insights for heatmap visualization.

```python
# Drop categorical columns for numerical analysis
numerical_data = df_data.drop(columns=["Channel", "Region"], errors='ignore')

# Generate correlation heatmap
plt.figure(figsize=(10, 6))
corr = numerical_data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Wholesale Customer Features")
plt.show()
```

**Key Findings from Correlation Analysis:**
- Strong correlation between Grocery and Detergents_Paper spending
- Notable correlation between Milk and Grocery purchases  
- Significant correlation between Milk and Detergents_Paper spending

These correlations suggest logical purchasing patterns where customers who buy more groceries also tend to purchase more household items and dairy products.

### Distribution Analysis

```python
# Generate pairplot for feature relationships
sns.pairplot(numerical_data)
plt.suptitle("Pairplot of Wholesale Customer Features", y=1.02)
plt.show()
```

The pairplot reveals that most data points cluster near the origin, indicating that the majority of customers have relatively low spending across most categories. However, there are clear outliers representing high-spending customers in specific categories.

### Summary Statistics

```python
# Calculate and visualize summary statistics
summary_stats = numerical_data.agg(['mean', 'min', 'max']).T

summary_stats.plot(kind='bar', figsize=(12, 6))
plt.title("Mean, Min, and Max for Each Feature")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.legend(title="Statistic")
plt.tight_layout()
plt.show()
```

**Key Observations:**
- Minimum values for all categories are at or near zero
- Fresh products show the highest average spending among all categories
- High variance in spending patterns across different product categories

## Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Remove categorical columns
X = df_data.drop(columns=['Channel', 'Region'], errors='ignore')

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Feature standardization is crucial for clustering algorithms as it ensures all variables contribute equally to distance calculations.

## KMeans Clustering Model

```python
from sklearn.cluster import KMeans

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataset
df_data['KMeans_Cluster'] = kmeans_labels
```

## Hierarchical Clustering Model

```python
from sklearn.cluster import AgglomerativeClustering

# Apply Agglomerative (Hierarchical) clustering
hierarchical = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
hc_labels = hierarchical.fit_predict(X_scaled)

# Add cluster labels to original dataset
df_data['Hierarchical_Cluster'] = hc_labels
```

### Cluster Visualization

```python
# Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create side-by-side comparison plots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# KMeans visualization
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, 
                palette='Set2', ax=axs[0])
axs[0].set_title("KMeans Clusters (PCA Projection)")
axs[0].set_xlabel("First Principal Component")
axs[0].set_ylabel("Second Principal Component")

# Hierarchical clustering visualization
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=hc_labels, 
                palette='Set1', ax=axs[1])
axs[1].set_title("Hierarchical Clusters (PCA Projection)")
axs[1].set_xlabel("First Principal Component")
axs[1].set_ylabel("Second Principal Component")

plt.tight_layout()
plt.show()
```

Both clustering methods show similar overall patterns with three distinct clusters. However, there are notable differences in boundary assignments, particularly for data points in overlapping regions.

## Model Optimization

### Elbow Method for KMeans

```python
# Determine optimal number of clusters using elbow method
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    kmeans_temp.fit(X_scaled)
    inertia.append(kmeans_temp.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linewidth=2, markersize=8)
plt.title("Elbow Method for Optimal K (KMeans)")
plt.xlabel("Number of Clusters")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.xticks(K_range)
plt.grid(True, alpha=0.3)
plt.show()
```

The elbow method suggests that the optimal number of clusters is around 3-4, where the rate of decrease in inertia begins to level off.

### Dendrogram for Hierarchical Clustering

```python
import scipy.cluster.hierarchy as sch

# Generate dendrogram
plt.figure(figsize=(15, 8))
dendrogram = sch.dendrogram(
    sch.linkage(X_scaled, method='ward'),
    no_labels=True,
    color_threshold=0.7*max(sch.linkage(X_scaled, method='ward')[:,2])
)
plt.title("Dendrogram (Ward's Linkage Method)", fontsize=16)
plt.ylabel("Euclidean Distance", fontsize=12)
plt.xlabel("Data Points", fontsize=12)
plt.axhline(y=0.7*max(sch.linkage(X_scaled, method='ward')[:,2]), 
            color='red', linestyle='--', alpha=0.7, label='Cut-off line')
plt.legend()
plt.tight_layout()
plt.show()
```

The dendrogram shows tight clustering at lower levels, indicating homogeneous groups within the dataset. The more dispersed merges at higher levels suggest the presence of distinct customer segments.

## Results and Analysis

### Silhouette Score Comparison

```python
from sklearn.metrics import silhouette_score

# Compare clustering methods across different numbers of clusters
cluster_range = range(2, 11)
silhouette_scores_kmeans = []
silhouette_scores_hierarchical = []

for k in cluster_range:
    # KMeans
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    kmeans_labels_temp = kmeans_temp.fit_predict(X_scaled)
    sil_score_km = silhouette_score(X_scaled, kmeans_labels_temp)
    silhouette_scores_kmeans.append(sil_score_km)
    
    # Hierarchical Clustering
    hc_temp = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
    hc_labels_temp = hc_temp.fit_predict(X_scaled)
    sil_score_hc = silhouette_score(X_scaled, hc_labels_temp)
    silhouette_scores_hierarchical.append(sil_score_hc)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores_kmeans, marker='o', 
         label='KMeans', linewidth=2, markersize=8)
plt.plot(cluster_range, silhouette_scores_hierarchical, marker='s', 
         label='Hierarchical', linewidth=2, markersize=8)

plt.title("Silhouette Score Comparison")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print optimal results
optimal_k_kmeans = cluster_range[np.argmax(silhouette_scores_kmeans)]
optimal_k_hierarchical = cluster_range[np.argmax(silhouette_scores_hierarchical)]

print(f"Optimal clusters for KMeans: {optimal_k_kmeans} (Score: {max(silhouette_scores_kmeans):.4f})")
print(f"Optimal clusters for Hierarchical: {optimal_k_hierarchical} (Score: {max(silhouette_scores_hierarchical):.4f})")
```

### Performance Analysis

**Key Findings:**
- Silhouette scores for both methods range between 0.3-0.6
- Hierarchical clustering performs exceptionally well with 2 clusters
- KMeans shows more consistent performance across different cluster numbers
- Both methods achieve reasonable clustering quality scores > 0.5 

## Citation

Cardoso, M. (2013). Wholesale customers [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5030X

**Additional Data Source:** [Kaggle - Wholesale Customers Dataset](https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set/data)

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# read dataset
columns = [
    "Recency (months)",
    "Frequency (times)",
    "Monetary (c.c. blood)",
    "Time (months)",
    "Target",  # "whether he/she donated blood in March 2007"
]
data = pd.read_csv("transfusion.data", names=columns, skiprows=1)

# drop target column to get only features
features = data.drop(columns=["Target"])

# convert to pytorch tensors
feature_tensor = torch.tensor(features.values, dtype=torch.float32)

# min-max scaling normalization
feature_min_values = feature_tensor.min(dim=0, keepdim=True).values
feature_max_values = feature_tensor.max(dim=0, keepdim=True).values
# features will have values range from 0 to 1
features_normalized = (feature_tensor - feature_min_values) / (
    feature_max_values - feature_min_values
)


# K-Means function
def k_means(normalized_features, clusters, iterations=50):
    # initialize centroids randomly
    centroids = normalized_features[
        torch.randperm(normalized_features.size(0))[:clusters]
    ]

    for _ in range(iterations):
        # calculate distances from data points to centroids
        distances = torch.cdist(normalized_features, centroids)

        # assign data point to nearest centroid
        labels = torch.argmin(distances, dim=1)

        for i in range(clusters):
            if torch.sum(labels == i) > 0:
                # assign centroid to the mean of data points and update centroid
                centroids[i] = normalized_features[labels == i].mean(dim=0)

    return labels, centroids


# Compute Silhouette Scores from 2 to 9 to find the optimal cluster K
silhouette_scores = []
for k in range(2, 10):
    cluster_labels, _ = k_means(features_normalized, k, 50)
    # use silhouette_score func from scikit-learn
    score = silhouette_score(features_normalized.numpy(), cluster_labels.numpy())
    silhouette_scores.append(score)


# Optimal number of clusters will have highest silhouette score
# get index and plus 2 because we start at index 0 with value K=2
optimal_cluster = silhouette_scores.index(max(silhouette_scores)) + 2

# Silhouette score plot
plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), silhouette_scores, "rx-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. Number of Clusters K")
plt.show()
print(f"Optimal number of cluster is: {optimal_cluster}\n")

# run k-means algorithm with optimal cluster
cluster_labels, centroids = k_means(features_normalized, optimal_cluster)

# Add column Cluster to the original dataset
data["Cluster"] = cluster_labels.numpy()


# Visualization
print("Some diagrams to help to understand the data and the clusters.\n")
# scatter plots for features pair
feature_pairs = [
    ("Recency (months)", "Frequency (times)"),
    ("Time (months)", "Recency (months)"),
    ("Time (months)", "Monetary (c.c. blood)"),
]

plt.figure(figsize=(15, 10))
for i, (x_feature, y_feature) in enumerate(feature_pairs, start=1):
    plt.subplot(2, 2, i)
    sns.scatterplot(
        x=features[x_feature],
        y=features[y_feature],
        hue=data["Cluster"],
        palette="viridis",
        alpha=0.7,
    )
    plt.title(f"{x_feature} vs {y_feature}")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend(title="Cluster")
plt.tight_layout()

# feature distributions by cluster
plt.figure(figsize=(15, 10))
# skip Target and Cluster column
for i, col in enumerate(data.columns[:-2], start=1):
    plt.subplot(2, 2, i)
    sns.boxplot(x="Cluster", y=col, data=data)
    plt.title(f"Distribution of {col} by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel(col)
plt.tight_layout()

# PCA - Principal Component Analysis
# visualize in 2D
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features_normalized.numpy())
reduced_centroids = pca.transform(centroids.numpy())
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=reduced_features[:, 0],
    y=reduced_features[:, 1],
    hue=cluster_labels.numpy(),
    palette="viridis",
    alpha=0.7,
)
plt.scatter(
    reduced_centroids[:, 0],
    reduced_centroids[:, 1],
    c="orange",
    marker="X",
    s=200,
    label="Centroids",
)
plt.title("PCA Visualization of Clusters")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Cluster")
plt.show()

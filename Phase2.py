import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTModel
from datasets import load_dataset
from sklearn.cluster import DBSCAN, MiniBatchKMeans, SpectralClustering
from sklearn.decomposition import PCA
from openTSNE import TSNE  # Faster t-SNE
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from PIL import Image
import os
import pandas as pd
from datetime import datetime
import community as community_louvain  

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CIFAR-10 dataset
# Dataset get downlaoded from internet using load_dataset function
# if want to use any local dataset then we can put that dataset into dataset directory provided and add a path below
dataset = load_dataset("fashion_mnist")   # To use cifar10 write cifar10
images = dataset["train"]["image"][:9000]  # Taken Subset of images for faster execution
labels = np.array(dataset["train"]["label"][:9000])

# Convert grayscale images to RGB and then to numpy arrays
images_rgb = [np.array(Image.fromarray(np.array(img)).convert("RGB")) for img in images]  # Comment it when using cifar10

# Load Vision Transformer Model & Image Processor
processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16').to(device)

# Batch feature extraction
batch_size = 64
features = []


# When using cifar10 change images_rgb to images
for i in range(0, len(images_rgb), batch_size):
    batch_images = images_rgb[i:i + batch_size]
    inputs = processor(images=batch_images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    features.append(batch_features)

features = np.concatenate(features, axis=0)
print("Feature extraction complete. Shape:", features.shape)

# Apply PCA for dimensionality reduction
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=50, svd_solver='randomized')
X_pca = pca.fit_transform(features_scaled)
print(f"PCA reduced shape: {X_pca.shape}")

# Apply fast t-SNE for dimensionality reduction
tsne = TSNE(n_jobs=-1, random_state=42, perplexity=30)
X_tsne = tsne.fit(X_pca)
print("t-SNE reduction complete.")

# Split data into training and testing sets
num_train = int(0.7 * len(X_tsne))
X_train, X_test = X_tsne[:num_train], X_tsne[num_train:]
y_train, y_test = labels[:num_train], labels[num_train:]

# DBSCAN Clustering
neighbors = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(X_train)
distances, _ = neighbors.kneighbors(X_train)
distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 6))
plt.plot(distances)
plt.title('k-distance Graph (5th Nearest Neighbor)')
plt.xlabel('Points sorted by distance')
plt.ylabel('Distance to 5th nearest neighbor')
plt.grid()
plt.show()

eps = 2.9 # Adjusted based on the elbow point on k-distance graph , set eps = 3.7 when using cifar10 dataset 
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
dbscan.fit(X_train)
y_pred_train = dbscan.labels_
y_pred_test = dbscan.fit_predict(X_test)

# Map clusters to labels
def map_clusters_to_labels(y_pred, y_true):
    cluster_mapping = {}
    for cluster in np.unique(y_pred):
        if cluster == -1:  # Skip noise cluster
            continue
        indices = np.where(y_pred == cluster)[0]
        true_labels = y_true[indices]
        if len(true_labels) > 0:
            most_common_label = np.bincount(true_labels).argmax()
            cluster_mapping[cluster] = most_common_label
    return np.array([cluster_mapping.get(cluster, -1) for cluster in y_pred])

mapped_labels = map_clusters_to_labels(y_pred_test, y_test)

# Calculate number of noise points found in dbscan
num_noise_points = np.sum(y_pred_test == -1)
print(f"Number of noise points in DBSCAN: {num_noise_points}")

# Louvain Clustering
def louvain_clustering(X):
    neighbors = NearestNeighbors(n_neighbors=10).fit(X)
    distances, indices = neighbors.kneighbors(X)
    graph = nx.Graph()

    for i in range(X.shape[0]):
        for j in indices[i]:
            weight = np.linalg.norm(X[i] - X[j])
            graph.add_edge(i, j, weight=weight)

    partition = community_louvain.best_partition(graph, weight='weight')
    return np.array([partition[i] for i in range(X.shape[0])])

y_pred_louvain = louvain_clustering(X_test)
mapped_labels_louvain = map_clusters_to_labels(y_pred_louvain, y_test)

# Ncut Clustering
def ncut_clustering(X, num_clusters=10):
    spectral = SpectralClustering(
        n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42, n_jobs=-1
    )
    return spectral.fit_predict(X)

y_pred_ncut = ncut_clustering(X_test, num_clusters=len(np.unique(y_test)))
mapped_labels_ncut = map_clusters_to_labels(y_pred_ncut, y_test)

# MiniBatch KMeans Clustering
def minibatch_kmeans_clustering(X, num_clusters):
    mb_kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=100)
    y_pred_kmeans = mb_kmeans.fit_predict(X)
    return y_pred_kmeans

y_pred_kmeans = minibatch_kmeans_clustering(X_test, num_clusters=len(np.unique(y_test)))
mapped_labels_kmeans = map_clusters_to_labels(y_pred_kmeans, y_test)


methods = ["DBSCAN", "Louvain", "Ncut", "MiniBatch KMeans"]
predictions = [mapped_labels, mapped_labels_louvain, mapped_labels_ncut, mapped_labels_kmeans]
true_labels = y_test  
num_datapoints = len(true_labels)  

# Collect evaluation metrics and confusion matrices 
metrics = []
confusion_matrices = {}

for method, y_pred in zip(methods, predictions):
    accuracy = accuracy_score(true_labels, y_pred)
    ari = adjusted_rand_score(true_labels, y_pred)
    nmi = normalized_mutual_info_score(true_labels, y_pred)
    cm = confusion_matrix(true_labels, y_pred)
    metrics.append((method, accuracy * 100, ari, nmi))
    confusion_matrices[method] = cm

# Convert metrics to a DataFrame
metrics_df = pd.DataFrame(metrics, columns=["Method", "Accuracy (%)", "ARI", "NMI"])

# Automatically get the desktop path
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# Generate a unique filename using timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_path = os.path.join(desktop_path, f"clustering_evaluation_metrics_{num_datapoints}_Testing_points_{timestamp}.png")

# Create a plot with metrics and confusion matrices
fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(8, 5 * (len(methods) + 1)))
fig.suptitle(f"Clustering Evaluation Metrics for {num_datapoints} testing examples", fontsize=14, fontweight="bold", y=0.92)

# Plot metrics table
axes[0].axis('tight')
axes[0].axis('off')
table = axes[0].table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(metrics_df.columns))))
axes[0].set_title("Evaluation Metrics", fontweight="bold", fontsize=12)

# Plot confusion matrices
for i, method in enumerate(methods, start=1):
    ax = axes[i]
    cm = confusion_matrices[method]
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_title(f"Confusion Matrix - {method}", fontweight="bold", fontsize=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))

    # Annotate the cells with numbers
    for (j, k), val in np.ndenumerate(cm):
        ax.text(k, j, f"{val}", ha='center', va='center', color="black", fontsize=9)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot as an image
plt.savefig(image_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Table and confusion matrices saved as image at: {image_path}")

# Generate a unique filename using timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_path = os.path.join(desktop_path, f"t-SNE_plots_for_{num_datapoints}_Testing_points_{timestamp}.png")

# t-SNE Plot (combined)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Move subtitle higher to avoid overlap with plots
fig.suptitle(f"t-SNE Plots for {num_datapoints} testing examples", 
             fontsize=16, fontweight="bold", y=0.95)  

# Plot True Labels
axes[0, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='tab10', alpha=0.7)
axes[0, 0].set_title("True Labels")
scatter_true = axes[0, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='tab10', alpha=0.7)
fig.colorbar(scatter_true, ax=axes[0, 0], label="True Labels")

# Plot DBSCAN Predicted Labels
noise_mask = y_pred_test == -1
scatter_dbscan = axes[0, 1].scatter(
    X_test[~noise_mask, 0], X_test[~noise_mask, 1],
    c=mapped_labels[~noise_mask], cmap='tab10', alpha=0.7, label="Clusters"
)
axes[0, 1].scatter(
    X_test[noise_mask, 0], X_test[noise_mask, 1],
    c='black', alpha=0.7, label="Noise"
)
axes[0, 1].set_title("DBSCAN Predicted Labels (Noise in Black)")
fig.colorbar(scatter_dbscan, ax=axes[0, 1], label="DBSCAN Clusters")
axes[0, 1].legend()

# Plot Louvain Predicted Labels
scatter_louvain = axes[0, 2].scatter(
    X_test[:, 0], X_test[:, 1],
    c=mapped_labels_louvain, cmap='tab10', alpha=0.7
)
axes[0, 2].set_title("Louvain Predicted Labels")
fig.colorbar(scatter_louvain, ax=axes[0, 2], label="Louvain Clusters")

# Plot MiniBatch KMeans Predicted Labels
scatter_kmeans = axes[1, 0].scatter(
    X_test[:, 0], X_test[:, 1],
    c=mapped_labels_kmeans, cmap='tab10', alpha=0.7
)
axes[1, 0].set_title("MiniBatch KMeans Predicted Labels")
fig.colorbar(scatter_kmeans, ax=axes[1, 0], label="KMeans Clusters")

# Plot Ncut Predicted Labels
scatter_ncut = axes[1, 1].scatter(
    X_test[:, 0], X_test[:, 1],
    c=mapped_labels_ncut, cmap='tab10', alpha=0.7
)
axes[1, 1].set_title("Ncut Predicted Labels")
fig.colorbar(scatter_ncut, ax=axes[1, 1], label="Ncut Clusters")

# Leave the last plot empty or add an annotation
axes[1, 2].axis('off')  # Turn off the last plot
axes[1, 2].text(0.5, 0.5, 'No Plot Available', fontsize=16, ha='center', va='center', color='gray')

# Adjust layout for better clarity
plt.tight_layout(rect=[0, 0, 1, 0.93])  
# Save the plot as an image
plt.savefig(image_path, dpi=300, bbox_inches='tight')
plt.show()

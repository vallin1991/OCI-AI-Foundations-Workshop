import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from ipywidgets import interact, IntSlider
from IPython.display import display, clear_output

#Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2] #We'll use only the first two features for simplicity

#Create a function for interactive clustering visualization
def plot_kmeans_clusters(num_clusters=2):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(X)

    #Create a scatter plot with different colors for each cluster
    plt.figure(figsize=(10, 6))
    for i in range(num_clusters):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i + 1}')
        plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s=200, c='black', marker='X', label='Centroids')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'K-Means Clustering (Number of Clusters:{num_clusters})')
        plt.legend()
        plt.grid(True)
        plt.show()

  #Create a slider for adjusting the number of clusters
num_clusters_slider = IntSlider(value=2, min=2, max=5, description='Number of Clusters')
#Create an interactive widget
interact(plot_kmeans_clusters, num_clusters=num_clusters_slider)

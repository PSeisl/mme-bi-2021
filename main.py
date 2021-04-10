# MME2: BII assignment 3 - k-means clustering
# SEISL Philipp (me20m003) & RIRSCH Karl-Philipp (me20m002)

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
from copy import deepcopy

# Read information like clusters, rows and columns
info = np.genfromtxt('input.csv', dtype=None, encoding='utf-8-sig', delimiter=';')
clusters = int(info[0][0])
rows = int(info[1][0])
cols = int(info[1][1])

# Read only the location of datapoints
df = pd.read_csv('input.csv', dtype=None, encoding='utf-8-sig', delimiter=';', decimal=',', skiprows=1)
kmeans = KMeans(clusters)
kmeans.fit(df)
df['cluster'] = kmeans.fit_predict(df)  # predict cluster belonging

data = df.values[:, 0:2]    # convert DataFrame to numpy-Array
cluster = df.values[:, 2]   # save cluster variable

# Generate Random starting points for centroids
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
print(mean, std)
centers = np.random.randn(clusters, cols)*std + mean



# Iterate Centroids
centers_old = np.zeros(centers.shape)  # to store old centers
centers_new = deepcopy(centers)  # Store new centers

data.shape
clusters0 = np.zeros(rows)
distances = np.zeros((rows, clusters))

error = np.linalg.norm(centers_new - centers_old)

# When, after an update, the estimate of that center stays the same, exit loop
while error != 0:
    # Measure the distance to every center
    for i in range(clusters):
        distances[:, i] = np.linalg.norm(data - centers_new[i], axis=1)
    # Assign all training data to closest center
    clusters0 = np.argmin(distances, axis=1)

    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(clusters):
        centers_new[i] = np.mean(data[clusters0 == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)
centers = deepcopy(centers_new)

# Create a Scatter-Plot
plt.scatter(data[:, 0], data[:, 1], c=cluster, cmap='rainbow', s=10)
plt.scatter(centers[:, 0], centers[:, 1], marker='d', c='grey', s=50)
plt.show()
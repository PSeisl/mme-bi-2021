# MME2: BI Assignment 3 - K-Means Clustering
# SEISL Philipp (me20m003) & RIRSCH Karl-Philipp (me20m002)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from scipy.spatial.distance import cdist
import csv

# Read information like clusters, rows and columns
info = np.genfromtxt('input.csv', dtype=None, encoding='utf-8-sig', delimiter=';')
clusters = int(info[0][0])  # Get cluster count
rows = int(info[1][0])      # Get row count
cols = int(info[1][1])      # Get column count

# Read only the location of datapoints
df = pd.read_csv('input.csv', dtype=None, encoding='utf-8-sig', delimiter=';', decimal=',', skiprows=1)
df['cluster'] = 0   # Add column to data frame for cluster assignment with prediction

data = df.values[:, 0:2]    # save coordinates from dataframe to Numpy-compatible array
cluster = df.values[:, 2]   # save cluster assignment variables Numpy-compatible array

# Generate Random starting points for centroids
mean = np.mean(data, axis=0)    # Calculate the mean value of coordinates
std = np.std(data, axis=0)      # Calculate the standard deviation of coordinates
centers = np.random.randn(clusters, cols)*std + mean    # Create random centroids by amount of clusters with boundaries

# Initialization for Centroid iteration loop
centers_old = np.zeros(centers.shape)   # Create Zero-Matrix to store old center coordinates
distances = np.zeros((rows, clusters))  # Create Zero-Matrix to store distances
error = np.linalg.norm(centers - centers_old)   # Calculate Matrix-norm as error indicator
iterations = 0

# Move centers until they stay in place, then exit the loop
while error != 0:
    distances = cdist(data, centers, metric='cityblock')    # Measure the distance to centers using Manhattan distance
    cluster = np.argmin(distances, axis=1)  # Assign data points to closest center/cluster
    centers_old = deepcopy(centers)     # Store old center coordinates
    for i in range(clusters):
        centers[i] = np.mean(data[cluster == i], axis=0)    # Calculate cluster means and update centers
    error = np.linalg.norm(centers - centers_old)   # Calculate new error value
    iterations += 1     # Count iterations

df['cluster'] = cluster     # Save new Cluster assignments to DataFrame

# Generate Scatter-Plot
plt.scatter(data[:, 0], data[:, 1], c=cluster, cmap='rainbow', s=10)    # Data points
plt.scatter(centers[:, 0], centers[:, 1], marker='d', c='grey', s=50)   # Centroids
plt.show()

# Save Data to Output CSV-file
df = df[['cluster', str(rows), str(cols)]]  # Rearrange DataFrame Columns
df_center = pd.DataFrame(centers)   # Convert Numpy-Array do Pandas DataFrame for easier CSV export

with open('output.csv', 'w+', newline='') as output:
    writer = csv.writer(output, delimiter=';')
    writer.writerow(str(iterations))
    df_center.to_csv(output, sep=';', mode='a', index=False, header=False)
    writer.writerow(str(clusters))
    writer.writerow([str(rows), str(cols)])
    df.to_csv(output, sep=';', mode='a', index=False, header=False)

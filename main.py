# MME2: BII assignment 3 - k-means clustering
# SEISL Philipp (me20m003) & RIRSCH Karl-Philipp (me20m002)


import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd

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

# Create a Scatter-Plot
print(cluster)
x_data = data[:, 0]
y_data = data[:, 1]
plt.scatter(x_data, y_data, c=cluster, cmap='rainbow')
plt.show()

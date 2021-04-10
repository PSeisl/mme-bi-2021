# MME2: BII assignment 3 - k-means clustering
# SEISL Philipp (me20m003) & RIRSCH Karl-Philipp (me20m002)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd

# Read data information like clusters, rows and columns
data = np.genfromtxt('input.csv', dtype=None, encoding='utf-8-sig', delimiter=';')
cluster = data[0][0]
rows = data[1][0]
cols = data[1][1]

# Read only the location of datapoints
value_data = pd.read_csv('input.csv', dtype=None, encoding='utf-8-sig', delimiter=';', decimal=',', skiprows=1)
x_data = value_data.iloc[:,0]
y_data = value_data.iloc[:,1]

# Create a Scatterplot
plt.scatter(x_data, y_data)
plt.show()


# MME2: BII assignment 3 - k-means clustering
# SEISL Philipp (me20m003) & RIRSCH Karl-Philipp (me20m002)

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
from copy import deepcopy

data = np.genfromtxt('input.csv', dtype=None, encoding='utf-8-sig', delimiter=';')
#for x in range(2,int(len(data))):
 #   data[x] = np.char.replace(data[x], ',', '.')
  #  data[x] = data[x].astype(np.float32)

print(data)

cluster = data[0][0]
rows = data[1][0]
cols = data[1][1]
print(cluster, rows, cols)

plt.scatter(data[:,0], data[0:1], s=7)

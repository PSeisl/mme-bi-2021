# MME2: BII assignment 3 - k-means clustering
# SEISL Philipp (me20m003) & RIRSCH Karl-Philipp (me20m002)

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd

data = np.genfromtxt('input.csv', dtype=None, encoding='utf-8-sig', delimiter=';')
cluster = data[0][0]
rows = data[1][0]
cols = data[1][1]
print(cluster, rows, cols)


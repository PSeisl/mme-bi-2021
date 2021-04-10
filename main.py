# MME2: BII assignment 3 - k-means clustering
# SEISL Philipp (me20m003) & RIRSCH Karl-Philipp (me20m002)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import xlrd
import csv
import pandas as pd

data = np.genfromtxt('input.csv', dtype=None, encoding='utf-8-sig', delimiter=';')


print(data)

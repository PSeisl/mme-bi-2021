# MME2: BII assignment 3 - k-means clustering
# SEISL Philipp (me20m003) & RIRSCH Karl-Philipp (me20m002)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import xlrd
import pandas as pd

excel_werte = pd.read_csv("input.csv", sep=";") #, skiprows=2
#excel_werte = np.genfromtxt('input.csv', delimiter=';')


print(excel_werte)

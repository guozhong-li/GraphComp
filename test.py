from skimage.segmentation import felzenszwalb
from skimage import graph
from skimage.util import img_as_float
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

M = 855
N = 1215
ds = np.fromfile('../sample_t2.dat', dtype=np.float32).reshape(-1, M, N)
print(ds.shape)
print(ds[0,0,0,])
print("数据类型:", ds.dtype)

reconstructed = np.load('./reconstructed_matrix_s500s1m100_5decimal.npy').reshape(-1, M, N)
print(reconstructed.shape)

print(reconstructed[0,0,0])
print("数据类型:", reconstructed.dtype)
import numpy as np
import graph_tool.all as gt
g = gt.Graph(10, directed=True)
vprop = g.new_vp("vector<double>")
for v in g.vertices():
    vprop[v] = [1.0, 2.0, 3.0]
arr = vprop.get_2d_array(range(3))
print(arr.shape)
print(arr)

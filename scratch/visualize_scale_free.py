import sys
import os

sys.path.append("/home/francesco/Universita/PhD/PROJECTS/InferenceBio")

import numpy as np
import graph_tool.all as gt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from inference_pipeline.data_generation.network_generator import generate_scale_free_W

if __name__ == "__main__":
    N = 10
    
    W = generate_scale_free_W(N, m=2, seed=42)
    
    g = gt.Graph(directed=True)
    g.add_vertex(N)
    
    e_color = g.new_edge_property("vector<double>")
    e_pen_width = g.new_edge_property("double")
    
    # Use coolwarm colormap
    cmap = cm.get_cmap("coolwarm")
    
    # Max possible range of generated weights is [-2.0, 2.0]
    norm = mcolors.Normalize(vmin=-2.0, vmax=2.0)
    
    for tgt in range(N):
        for src in range(N):
            weight = W[tgt, src]
            if weight != 0:
                e = g.add_edge(src, tgt)
                
                # Assign color based on weight using coolwarm palette
                rgba = cmap(norm(weight))
                e_color[e] = list(rgba)
                
                # Make width proportional to absolute weight, and generally much wider
                # Since abs(weight) is between 0.5 and 2.0, this yields widths between 5.0 and 20.0
                e_pen_width[e] = abs(weight) * 10.0
    
    output_path = "/home/francesco/Universita/PhD/PROJECTS/InferenceBio/scratch/network_visualization.png"
    
    gt.graph_draw(
        g,
        vertex_size=40,
        vertex_fill_color="white",
        edge_color=e_color,
        edge_pen_width=e_pen_width,
        output_size=(600, 600),
        output=output_path
    )
    print(f"Visualization saved to {output_path}")

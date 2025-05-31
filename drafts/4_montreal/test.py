# render_dag.py
import numpy as np
from graphviz import Digraph

# —— User params ——
node_names = ["x-pos", "y-pos", "shape", "color", "orientation", "scale"]
adj_matrix = np.array([
    # x     y     sh    co    or    sc
    [0.0,  0.40, 0.00, 0.00, 0.34, 0.00],  # x-pos
    [0.00, 0.00, 0.38, 0.00, 0.00, 0.00],  # y-pos
    [0.00, 0.00, 0.00, 0.42, 0.00, 0.00],  # shape
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # color
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # orientation
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # scale
])

# —— Graphviz setup ——
g = Digraph('G', filename='dag', format='png')
g.attr(rankdir='LR', splines='ortho', nodesep='0.6', ranksep='0.8')
g.attr('node', shape='circle', style='filled', fillcolor='#AED6F1', fontname='Helvetica', fontsize='12')
g.attr('edge', arrowhead='vee', arrowsize='0.7', penwidth='1.5')

# Add nodes
for i, name in enumerate(node_names):
    g.node(str(i), label=name)

# Add edges with weights
n = adj_matrix.shape[0]
for i in range(n):
    for j in range(n):
        w = adj_matrix[i, j]
        if w > 0:
            # edge label and thickness
            g.edge(str(i), str(j),
                   label=f"{w:.2f}",
                   fontname='Helvetica', fontsize='10',
                   penwidth=str(1 + 4*w))  # thicker lines for larger weights

# Render to PNG
g.render(cleanup=True)
print("Wrote dag.png")

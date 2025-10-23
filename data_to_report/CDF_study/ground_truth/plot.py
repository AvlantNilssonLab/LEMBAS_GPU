import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load CSV with no header (each row is an edge)
df = pd.read_csv("LEMBAS/data_to_report/CDF_study/ground_truth/ground_truth_ligands.txt", header=None)
edges = [tuple(row) for row in df.values]
edges = [tuple(row) for row in df.values]

source_nodes = edges[0]
target_nodes = edges[1]

edges = list(zip(source_nodes, target_nodes))
# Create directed graph
G = nx.DiGraph()
G.add_edges_from(edges)

if 1==0:
    # Plot graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray',
            node_size=1200, arrowsize=20)
    plt.title("Graph from CSV Edges")
    plt.tight_layout()
    plt.show()

# Plot histogram of node degree (you can toggle in_degree or out_degree if needed)
degrees = [G.degree(n) for n in G.nodes()]  # use G.in_degree(n) or G.out_degree(n) if needed

plt.figure(figsize=(8, 6))
plt.hist(degrees, bins=range(1, max(degrees)+2), align='left', color='skyblue', edgecolor='black')
plt.xlabel("Node Degree")
plt.ylabel("Frequency")
plt.title("Histogram of Node Degrees")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
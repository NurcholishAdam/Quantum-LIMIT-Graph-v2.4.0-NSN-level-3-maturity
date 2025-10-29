"""
Quantum-LIMIT-Graph v2.4.0-NSN Core Module
Provides graph loading, benchmarking, and visualization functions
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

def load_graph(graph_path: Optional[str] = None, graph_type: str = "quantum") -> nx.Graph:
    """
    Load or create a quantum LIMIT graph
    
    Args:
        graph_path: Path to saved graph file (optional)
        graph_type: Type of graph to create if no path provided
        
    Returns:
        NetworkX graph object
    """
    if graph_path:
        try:
            # Try to load existing graph
            G = nx.read_gpickle(graph_path)
            print(f"Loaded graph from {graph_path}")
            return G
        except Exception as e:
            print(f"Could not load graph: {e}")
    
    # Create default quantum LIMIT graph
    print(f"Creating new {graph_type} graph...")
    
    if graph_type == "quantum":
        # Create quantum entanglement graph structure
        G = nx.Graph()
        
        # Add quantum nodes (qubits)
        num_qubits = 10
        for i in range(num_qubits):
            G.add_node(f"Q{i}", 
                      qubit_id=i,
                      state="superposition",
                      fidelity=0.89 + (i % 3) * 0.02)
        
        # Add entanglement edges
        for i in range(num_qubits - 1):
            G.add_edge(f"Q{i}", f"Q{i+1}", 
                      weight=0.85 + (i % 4) * 0.03,
                      entanglement_type="bell_state")
        
        # Add some long-range entanglements
        G.add_edge("Q0", "Q5", weight=0.78, entanglement_type="ghz_state")
        G.add_edge("Q2", "Q7", weight=0.81, entanglement_type="w_state")
        
    elif graph_type == "edit":
        # Create edit provenance graph
        G = nx.DiGraph()
        
        edits = [
            ("edit_1", {"language": "english", "success": True, "rank": 128}),
            ("edit_2", {"language": "russian", "success": True, "rank": 64}),
            ("edit_3", {"language": "spanish", "success": True, "rank": 256}),
            ("edit_4", {"language": "chinese", "success": False, "rank": 128}),
        ]
        
        for edit_id, attrs in edits:
            G.add_node(edit_id, **attrs)
        
        # Add edit dependencies
        G.add_edge("edit_1", "edit_2", dependency_type="sequential")
        G.add_edge("edit_2", "edit_3", dependency_type="parallel")
        G.add_edge("edit_1", "edit_4", dependency_type="override")
        
    else:
        # Create general graph
        G = nx.karate_club_graph()
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def benchmark_query(G: nx.Graph, 
                   query_type: str = "shortest_path",
                   source: Optional[str] = None,
                   target: Optional[str] = None,
                   num_iterations: int = 100) -> Dict:
    """
    Benchmark graph query performance
    
    Args:
        G: NetworkX graph
        query_type: Type of query to benchmark
        source: Source node for path queries
        target: Target node for path queries
        num_iterations: Number of iterations for timing
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"Benchmarking {query_type} query...")
    
    nodes = list(G.nodes())
    if not source and len(nodes) > 0:
        source = nodes[0]
    if not target and len(nodes) > 1:
        target = nodes[-1]
    
    results = {
        "query_type": query_type,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "iterations": num_iterations,
    }
    
    try:
        start_time = time.time()
        
        if query_type == "shortest_path":
            for _ in range(num_iterations):
                if nx.has_path(G, source, target):
                    path = nx.shortest_path(G, source, target)
            results["path_length"] = len(path) if 'path' in locals() else None
            
        elif query_type == "centrality":
            for _ in range(num_iterations):
                centrality = nx.degree_centrality(G)
            results["most_central"] = max(centrality, key=centrality.get)
            results["max_centrality"] = max(centrality.values())
            
        elif query_type == "clustering":
            for _ in range(num_iterations):
                clustering = nx.clustering(G)
            results["avg_clustering"] = sum(clustering.values()) / len(clustering)
            
        elif query_type == "pagerank":
            for _ in range(num_iterations):
                pagerank = nx.pagerank(G)
            results["top_node"] = max(pagerank, key=pagerank.get)
            results["max_pagerank"] = max(pagerank.values())
            
        else:
            raise ValueError(f"Unknown query type: {query_type}")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        results["total_time_sec"] = elapsed
        results["avg_time_ms"] = (elapsed / num_iterations) * 1000
        results["queries_per_sec"] = num_iterations / elapsed
        results["success"] = True
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
    
    print(f"Benchmark complete: {results.get('avg_time_ms', 'N/A'):.2f} ms per query")
    return results


def visualize_subgraph(G: nx.Graph, 
                      center_node: Optional[str] = None,
                      depth: int = 2,
                      layout: str = "spring",
                      title: str = "Quantum LIMIT Graph") -> plt.Figure:
    """
    Visualize a subgraph centered on a specific node
    
    Args:
        G: NetworkX graph
        center_node: Node to center the subgraph on
        depth: How many hops from center to include
        layout: Layout algorithm (spring, circular, kamada_kawai)
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    print(f"Visualizing subgraph (depth={depth})...")
    
    # Select subgraph
    if center_node and center_node in G:
        # BFS to find nodes within depth
        if isinstance(G, nx.DiGraph):
            subgraph_nodes = set([center_node])
            for _ in range(depth):
                neighbors = set()
                for node in subgraph_nodes:
                    neighbors.update(G.successors(node))
                    neighbors.update(G.predecessors(node))
                subgraph_nodes.update(neighbors)
        else:
            subgraph_nodes = nx.single_source_shortest_path_length(G, center_node, cutoff=depth).keys()
        
        H = G.subgraph(subgraph_nodes).copy()
    else:
        # Use full graph or sample
        if G.number_of_nodes() > 50:
            # Sample random subgraph if too large
            sample_nodes = list(G.nodes())[:50]
            H = G.subgraph(sample_nodes).copy()
        else:
            H = G.copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(H, k=1, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(H)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(H)
    else:
        pos = nx.spring_layout(H)
    
    # Node colors based on attributes
    node_colors = []
    for node in H.nodes():
        attrs = H.nodes[node]
        if 'fidelity' in attrs:
            # Color by fidelity (quantum nodes)
            node_colors.append(attrs['fidelity'])
        elif 'success' in attrs:
            # Color by success (edit nodes)
            node_colors.append(1.0 if attrs['success'] else 0.3)
        else:
            node_colors.append(0.7)
    
    # Draw graph
    nx.draw_networkx_nodes(H, pos, 
                          node_color=node_colors,
                          node_size=500,
                          cmap=plt.cm.RdYlGn,
                          vmin=0, vmax=1,
                          ax=ax)
    
    nx.draw_networkx_edges(H, pos,
                          edge_color='gray',
                          alpha=0.5,
                          arrows=isinstance(H, nx.DiGraph),
                          ax=ax)
    
    nx.draw_networkx_labels(H, pos,
                           font_size=8,
                           font_color='black',
                           ax=ax)
    
    # Add edge labels for weights
    edge_labels = {}
    for u, v, data in H.edges(data=True):
        if 'weight' in data:
            edge_labels[(u, v)] = f"{data['weight']:.2f}"
    
    nx.draw_networkx_edge_labels(H, pos, edge_labels, font_size=6, ax=ax)
    
    ax.set_title(f"{title}\n({H.number_of_nodes()} nodes, {H.number_of_edges()} edges)")
    ax.axis('off')
    
    plt.tight_layout()
    print("Visualization complete")
    return fig


# Additional utility functions

def get_graph_stats(G: nx.Graph) -> Dict:
    """Get statistical summary of graph"""
    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_directed": isinstance(G, nx.DiGraph),
    }
    
    if G.number_of_nodes() > 0:
        if nx.is_connected(G) if not isinstance(G, nx.DiGraph) else nx.is_weakly_connected(G):
            stats["is_connected"] = True
            stats["diameter"] = nx.diameter(G) if not isinstance(G, nx.DiGraph) else None
        else:
            stats["is_connected"] = False
            stats["num_components"] = nx.number_connected_components(G) if not isinstance(G, nx.DiGraph) else nx.number_weakly_connected_components(G)
    
    return stats


def save_graph(G: nx.Graph, filepath: str) -> bool:
    """Save graph to file"""
    try:
        nx.write_gpickle(G, filepath)
        print(f"Graph saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving graph: {e}")
        return False
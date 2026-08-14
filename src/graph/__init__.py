from .grid_graph import build_grid_graph
try:
    from .gnn_features import prepare_gnn_graph
except ImportError:
    pass

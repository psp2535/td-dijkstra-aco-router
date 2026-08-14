import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Fix random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class IMVRMGraphConv(nn.Module):
    """
    Custom Graph Convolution layer matching the GNN design in the IM-VRM paper.
    """
    def __init__(self, in_features, out_features):
        super(IMVRMGraphConv, self).__init__()
        self.w_self = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.w_neigh = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        # Kaiming uniform weight initialization
        nn.init.kaiming_uniform_(self.w_self, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.w_neigh, a=np.sqrt(5))
        # Bias initialization
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_self)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x, adj):
        """
        Forward pass separating Aggregation (Eq. 26) and Combination (Eq. 27).
        x: Node features [N, in_features]
        adj: Binary adjacency matrix [N, N]
        """
        # ==========================================
        # 1. AGGREGATE Step (Eq. 26): h_v^(l+1) = AGGREGATE({h_u^(l) | u in N(v)})
        # We perform row-normalized mean aggregation over neighbors.
        # ==========================================
        # Calculate out-degrees
        degrees = adj.sum(dim=1, keepdim=True)
        # Prevent division by zero for isolated nodes
        degrees = torch.clamp(degrees, min=1.0)
        # Normalized adjacency matrix D^-1 * A
        norm_adj = adj / degrees
        # Aggregate neighbor representations
        aggregated = torch.matmul(norm_adj, x) # [N, in_features]
        
        # ==========================================
        # 2. COMBINE Step (Eq. 27): h_v^(l+1) = COMBINE(h_v^(l), aggregated_neighbors)
        # Combine node's own state and neighbor representation via linear layer weights
        # ==========================================
        out = torch.matmul(x, self.w_self) + torch.matmul(aggregated, self.w_neigh) + self.bias
        return out

class IMVRMNodeEncoder(nn.Module):
    """
    Lightweight 2-layer GNN model to compute spatial node embeddings.
    Dimensions: Input (4) -> Hidden (32) -> Output (64)
    """
    def __init__(self, in_dim=4, hidden_dim=32, embed_dim=64):
        super(IMVRMNodeEncoder, self).__init__()
        self.conv1 = IMVRMGraphConv(in_dim, hidden_dim)
        self.conv2 = IMVRMGraphConv(hidden_dim, embed_dim)
        
    def forward(self, x, adj):
        # Layer 1
        h1 = self.conv1(x, adj)
        h1 = F.relu(h1)
        # Layer 2 (final embeddings layer: Embedding(v) = h_v^(theta))
        h2 = self.conv2(h1, adj)
        return h2

def train_gnn_embeddings(graph_data, epochs=100, lr=0.01, seed=42):
    """
    Train GNN parameters by minimizing distance reconstruction loss (Eq. 28).
    
    Loss equation (Eq. 28):
    Lf = sum_{(u,v) in E} ( d_pred(u,v) - d_true(u,v) )^2
    where d_pred(u,v) = ||h_u - h_v||_2
    """
    set_seed(seed)
    
    x = graph_data["x"]
    edge_index = graph_data["edge_index"]
    edge_attr = graph_data["edge_attr"]
    adj_matrix = graph_data["adj_matrix"]
    
    N = x.shape[0]
    E = edge_index.shape[1]
    
    # Initialize GNN model
    model = IMVRMNodeEncoder(in_dim=4, hidden_dim=32, embed_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Extract true transition distances (first feature of edge_attr)
    # We scale distances to kilometers (divide meters by 1000) for numeric stability
    d_true = edge_attr[:, 0] / 1000.0
    
    # Check if we have edges to run training
    if E == 0:
        # Dry run on empty edges
        model.eval()
        with torch.no_grad():
            embeddings = model(x, adj_matrix)
        return embeddings, []
        
    history = []
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass to generate node embeddings
        embeddings = model(x, adj_matrix) # [N, 64]
        
        # Extract source and target node embeddings for all edges
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        h_src = embeddings[src_nodes]
        h_dst = embeddings[dst_nodes]
        
        # Predict distance as L2 norm of embedding difference
        # Add epsilon to prevent gradient explosion when distance approaches zero
        d_pred = torch.sqrt(torch.sum((h_src - h_dst) ** 2, dim=1) + 1e-8)
        
        # Loss function (Eq. 28): Mean Squared Error of distance reconstruction
        loss = torch.mean((d_pred - d_true) ** 2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        loss_val = float(loss.item())
        history.append(loss_val)
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"  GNN Epoch {epoch:03d}/{epochs:03d} | Loss: {loss_val:.6f}")
            
    # Return trained embeddings and loss history
    model.eval()
    with torch.no_grad():
        final_embeddings = model(x, adj_matrix)
        
    return final_embeddings, history

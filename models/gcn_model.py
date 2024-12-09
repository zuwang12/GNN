import torch
import torch.nn.functional as F
import torch.nn as nn

from models.gcn_layers import ResidualGatedGCNLayer, MLP
from utils.model_utils import *


class ResidualGatedGCNModel(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.

    References:
        Paper: https://arxiv.org/pdf/1711.07553v2.pdf
        Code: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(self, config, dtypeFloat=torch.float32, dtypeLong=torch.int64):
        super(ResidualGatedGCNModel, self).__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeFloat = torch.float32 # test for debugging
        self.dtypeLong = dtypeLong
        # Define net parameters
        self.num_nodes = config.num_nodes
        self.node_dim = config.node_dim
        self.voc_nodes_in = config['voc_nodes_in']
        self.voc_nodes_out = config['num_nodes']  # config['voc_nodes_out']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.mlp_layers = config['mlp_layers']
        self.aggregation = config['aggregation']
        # Node and edge embedding layers/lookups
        self.nodes_coord_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim//2, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim//2)
        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)

    def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw):
        # Data type checks and conversions
        x_edges = torch.tensor(x_edges, dtype=self.dtypeLong) if isinstance(x_edges, np.ndarray) else x_edges
        x_edges_values = torch.tensor(x_edges_values, dtype=self.dtypeFloat) if isinstance(x_edges_values, np.ndarray) else x_edges_values
        x_nodes = torch.tensor(x_nodes, dtype=self.dtypeLong) if isinstance(x_nodes, np.ndarray) else x_nodes
        x_nodes_coord = torch.tensor(x_nodes_coord, dtype=self.dtypeFloat) if isinstance(x_nodes_coord, np.ndarray) else x_nodes_coord
        y_edges = torch.tensor(y_edges, dtype=self.dtypeLong) if isinstance(y_edges, np.ndarray) else y_edges
        edge_cw = torch.tensor(edge_cw, dtype=self.dtypeFloat) if isinstance(edge_cw, np.ndarray) else edge_cw
        
        # Node and edge embeddings
        x = self.nodes_coord_embedding(x_nodes_coord)
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))
        e_tags = self.edges_embedding(x_edges)
        e = torch.cat((e_vals, e_tags), dim=3)

        # GCN layers
        # for layer in self.gcn_layers:
        #     x, e = layer(x, e)
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e)

        # MLP classifier
        y_pred_edges = self.mlp_edges(e)
        
        # Compute loss
        loss = loss_edges(y_pred_edges, y_edges, edge_cw)
        
        return y_pred_edges, loss

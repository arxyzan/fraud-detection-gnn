import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv


class GAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Properties
        self.config = config
        self.num_heads = config.num_heads
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.dropout = config.dropout

        # Layers
        self.conv1 = GATConv(self.input_dim, self.hidden_dim, heads=self.num_heads)
        self.relu1 = nn.ReLU()
        self.conv2 = GATConv(self.num_heads * self.hidden_dim, self.hidden_dim, heads=self.num_heads)
        self.relu2 = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(self.num_heads * self.hidden_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu1(x)
        x = self.conv2(x, edge_index)
        x = self.relu2(x)
        x = self.classifier(x)
        return x
    

class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        # Properties
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim

        # Layers
        self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, 2)
        self.classifier = nn.Linear(2, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        embeddings = h.tanh()
        out = self.classifier(embeddings)
        return torch.sigmoid(out)


class GIN(nn.Module):
    def __init__(self, config):
        super(GIN, self).__init__()
        # Properties
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim

        # Layers
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
        )
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        ))
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return torch.sigmoid(x)

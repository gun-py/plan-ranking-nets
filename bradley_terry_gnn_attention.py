import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GraphConv, GATConv
from torch_geometric.data import Data

class BradleyTerryGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, temperature=1.0):
        super(BradleyTerryGNN, self).__init__()
        self.temperature = temperature
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
        self.fc = nn.Linear(output_dim, 1)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        scores = self.fc(x).squeeze()
        scores = F.softplus(scores)
        
        log_likelihood = 0
        for i in range(edge_index.size(1)):
            a = edge_index[0, i]
            b = edge_index[1, i]
            p = scores[a] / (scores[a] + scores[b])
            p = p ** (1.0 / self.temperature)
            
            log_likelihood += data.edge_attr[i] * torch.log(p) + (1 - data.edge_attr[i]) * torch.log(1 - p)
        
        return -log_likelihood

class BradleyTerryGNNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, temperature=1.0):
        super(BradleyTerryGNNWithAttention, self).__init__()
        self.temperature = temperature
        
        # attention layers
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)

        self.fc = nn.Linear(output_dim, 1)
        
    def forward(self, data, original_df):
        x, edge_index = data.x, data.edge_index
        
        # attention dec
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        scores = self.fc(x).squeeze()
        scores = F.softplus(scores)
        
        log_likelihood = 0
        for i, row in original_df.iterrows():
            a = int(row['A_idx'])
            b = int(row['B_idx'])
            p = scores[a] / (scores[a] + scores[b])
            p = p ** (1.0 / self.temperature)
            
            log_likelihood += row['Wins'] * torch.log(p) + (1 - row['Wins']) * torch.log(1 - p)
        
        return -log_likelihood

''''
trainer format:
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = model(graph_data)
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    optimizer.step()
    if verbose:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

model = BradleyTerryGNNWithAttention(input_dim, hidden_dim, output_dim, temperature=temperature)
optimizer = Adam(model.parameters(), lr=learning_rate)
'''

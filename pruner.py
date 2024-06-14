import torch
import torch.nn as nn

class BTPruner(nn.Module):
    def __init__(self, num_players, input_dim, hidden_dim, nhead, num_layers, temperature=1.0):
        super(BTPruner, self).__init__()
        self.temperature = temperature
    
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, 1)
    
        self.pos_encoder = nn.Parameter(torch.zeros(1, num_players, hidden_dim))
        
        self.pruning_threshold = 0.01  
        self.edge_pruning_threshold = 0.01  

    def forward(self, data, original_df):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        x += self.pos_encoder
        
        x = x.unsqueeze(1)  # [num_nodes, 1, hidden_dim]
        
        x = self.transformer(x)
        x = x.squeeze(1)
        
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
    
    def prune_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    mask = torch.abs(param) > self.pruning_threshold
                    param *= mask.float()
    
    def prune_edges(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if data.edge_attr is not None else torch.ones(edge_index.size(1))
        
        mask = edge_attr > self.edge_pruning_threshold
        pruned_edge_index = edge_index[:, mask]
        
        data.edge_index = pruned_edge_index
        return data

'''
Schema:
model.prune_weights()
graph_data = model.prune_edges(graph_data)
'''

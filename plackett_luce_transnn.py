import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, prune

class PlackettLuceAsTransformer(nn.Module):
    def __init__(self, num_players, input_dim, hidden_dim, nhead, num_layers, temperature=1.0):
        super(PlackettLuceAsTransformer, self).__init__()
        self.temperature = temperature
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, 1)
        
        self.pos_encoder = nn.Parameter(torch.zeros(1, num_players, hidden_dim))
        self.permutations = self.generate_permutations(num_players)
    
    def forward(self, data, original_df):
        x, edge_index = data.x, data.edge_index
 
        x = self.embedding(x)
        x += self.pos_encoder
        
        x = x.unsqueeze(1)
        
        x = self.transformer(x)  
        x = x.squeeze(1) 
        
        scores = self.fc(x).squeeze()
        scores = F.softplus(scores)
        
        log_likelihood = self.plackett_luce_likelihood(scores, original_df)
        
        return -log_likelihood
    
    def plackett_luce_likelihood(self, scores, original_df):
        log_likelihood = 0
        for perm in self.permutations:
            perm_likelihood = 1.0
            for i in range(len(perm)):
                for j in range(i + 1, len(perm)):
                    a_idx = perm[i]
                    b_idx = perm[j]
                    p = scores[a_idx] / (scores[a_idx] + scores[b_idx])
                    p = p ** (1.0 / self.temperature)  # Apply temperature scaling
                    if original_df.loc[a_idx, b_idx] == 1:
                        perm_likelihood *= p
                    else:
                        perm_likelihood *= (1 - p)
            log_likelihood += torch.log(perm_likelihood)
        return log_likelihood / len(self.permutations)
    
    def generate_permutations(self, num_players):
        from itertools import permutations
        perms = list(permutations(range(num_players)))
        return torch.tensor(perms, dtype=torch.long)
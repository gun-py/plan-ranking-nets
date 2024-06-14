import torch.nn as nn
from torch_geometric.nn import GATConv
from sklearn.cluster import DBSCAN

import torch.nn.functional as F
import torch


class EncoderMatcherGNN(nn.Module):
    def __init__(self, num_players, input_dim, hidden_dim, nhead, num_layers, temperature=1.0):
        super(EncoderMatcherGNN, self).__init__()
        self.temperature = temperature
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=nhead, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * nhead, hidden_dim, heads=nhead, dropout=0.2)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_matcher = nn.Linear(hidden_dim, 1)
        self.dbscan = DBSCAN(eps=0.5, min_samples=2)
        self.pos_encoder = nn.Parameter(torch.zeros(1, num_players, hidden_dim))
        
    def forward(self, data, original_df):
        x, edge_index = data.x, data.edge_index
        

        x = self.embedding(x)
        
        x += self.pos_encoder
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x_transformed = self.transformer(x.unsqueeze(1)).squeeze(1)
        scores_matcher = self.fc_matcher(x).squeeze()
        scores_matcher = F.softplus(scores_matcher)  
        
        clusters = self.dbscan.fit_predict(x.detach().numpy())
        
        # Plackett-Luce model
        log_likelihood_extractor = self.plackett_luce_likelihood(scores_matcher, original_df)
        
        return -log_likelihood_extractor
    
    def plackett_luce_likelihood(self, scores, original_df):
        log_likelihood = 0
        for perm in self.permutations:
            perm_likelihood = 1.0
            for i in range(len(perm)):
                for j in range(i + 1, len(perm)):
                    a_idx = perm[i]
                    b_idx = perm[j]
                    p = scores[a_idx] / (scores[a_idx] + scores[b_idx])
                    p = p ** (1.0 / self.temperature)  
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
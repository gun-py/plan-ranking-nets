import torch.nn as nn
import torch.nn.functional as F
import torch

class TransformerEncoder(nn.Module):
    def __init__(self, num_players, input_dim, hidden_dim, nhead, num_layers, temperature=1.0):
        super(TransformerEncoder, self).__init__()
        self.temperature = temperature
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.pos_encoder = nn.Parameter(torch.zeros(1, num_players, hidden_dim))
        
    def forward(self, data, original_df):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        x += self.pos_encoder

        x = x.unsqueeze(1) 
        
        x = self.transformer(x)  # [num_nodes, 1, hidden_dim]
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
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        return -log_likelihood


        
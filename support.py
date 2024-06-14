import torch
import torch.nn as nn
from torch.optim import LBFGS, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

class BradleyTerryDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'A_idx': int(row['A_idx']),
            'B_idx': int(row['B_idx']),
            'Wins': row['Wins']
        }

class BradleyTerryModel(nn.Module):
    def __init__(self, num_players, temperature=1.0):
        super(BradleyTerryModel, self).__init__()
        self.num_players = num_players
        self.temperature = temperature
        self.scores = nn.Parameter(torch.ones(num_players))

    def forward(self, data):
        log_likelihood = 0

        for i, match in enumerate(data):
            a = match['A_idx']
            b = match['B_idx']
            p = self.scores[a] / (self.scores[a] + self.scores[b])
            p = p ** (1.0 / self.temperature) 
            log_likelihood += match['Wins'] * torch.log(p) + (1 - match['Wins']) * torch.log(1 - p)

        return -log_likelihood

    def constraint(self):
        self.scores.data = torch.nn.functional.softplus(self.scores.data)

def base_bt_check_vanilla(df, model, optimizer, scheduler, num_epochs=100, batch_size=16, verbose=True):
    dataset = BradleyTerryDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            model.constraint()  # +ve
            total_loss += loss.item()

        scheduler.step() 
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')



'''
Trainer formats:
model = BradleyTerryModel(num_players, temperature=temperature)
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  
'''
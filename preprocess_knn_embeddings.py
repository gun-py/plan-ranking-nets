from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
import numpy as np

def create_graph_from_df(df, num_players):
    edges = torch.tensor([df['A_idx'].tolist(), df['B_idx'].tolist()], dtype=torch.long)
    edge_attr = torch.tensor(df['Wins'].tolist(), dtype=torch.float)
    x = torch.ones((num_players, 1), dtype=torch.float)
    graph_data = Data(x=x, edge_index=edges, edge_attr=edge_attr)
    return graph_data

def create_knn_graph_from_df_KNN_embedings(df, num_players, k):
    edges = torch.tensor([df['A_idx'].tolist(), df['B_idx'].tolist()], dtype=torch.long)
    adj = to_dense_adj(edges, max_num_nodes=num_players).squeeze()
    knn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    knn.fit(np.eye(num_players))
    knn_distances, knn_indices = knn.kneighbors(np.eye(num_players))
    knn_adj = torch.zeros_like(adj)
    for i in range(num_players):
        for j in knn_indices[i]:
            if i != j:
                knn_adj[i, j] = 1
                knn_adj[j, i] = 1
    
    edge_index, _ = dense_to_sparse(knn_adj)
    x = torch.ones((num_players, 1), dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=None)
    return graph_data, knn_indices
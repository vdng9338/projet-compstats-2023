import torch
import torch.nn as nn
from link_prediction.models import VGAE
from torch_geometric import datasets

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = datasets.Planetoid(root='__data/Cora', name='Cora', split='public')
    graph = dataset.edge_index
    model = VGAE(dataset.x.shape[1]).to(device)
    print(model(dataset.x.to(device), graph.to(device)))

if __name__ == '__main__':
    main()

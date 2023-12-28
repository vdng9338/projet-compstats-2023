import torch
from torch.optim import Adam
import torch.nn as nn
import torch_geometric
from torch_geometric import datasets
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models.autoencoder import VGAE

class TG_VGAEEncoder(nn.Module):
    def __init__(self, input_dim, interm_dim=32, latent_dim=16):
        super().__init__()
        self.conv1 = GCNConv(input_dim, interm_dim)
        self.relu = nn.ReLU()
        self.conv2_mu = GCNConv(interm_dim, latent_dim)
        self.conv2_logstd = GCNConv(interm_dim, latent_dim)
    
    def forward(self, X, edge_index):
        X = self.conv1(X, edge_index)
        X = self.relu(X)
        mu = self.conv2_mu(X, edge_index)
        logstd = self.conv2_logstd(X, edge_index)
        return mu, logstd

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = datasets.Planetoid(root='__data/Cora', name='Cora', split='public')
    x_dim = dataset[0].x.shape[1]
    num_nodes = dataset[0].x.shape[0]
    """train_data, val_edges, test_edges = train_val_test_split(dataset[0])
    neg_val = sample_negative_edges(dataset[0], val_edges.transpose(0, 1))
    neg_test = sample_negative_edges(dataset[0], test_edges.transpose(0, 1))"""
    transform = RandomLinkSplit(num_val=.05, num_test=.1, is_undirected=True, split_labels=True)
    train_data, val_data, test_data = transform(dataset[0])
    x = train_data.x.to(device)
    pos_edge_index = train_data.edge_index.to(device)
    pos_edge_label_index = train_data.pos_edge_label_index.to(device)
    neg_edge_label_index = train_data.neg_edge_label_index.to(device)
    encoder = TG_VGAEEncoder(x_dim).to(device)
    model = VGAE(encoder).to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=0.01)
    num_epochs = 200
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        z = model.encode(x, pos_edge_index)
        loss = model.recon_loss(z, pos_edge_index) + 1/num_nodes*model.kl_loss()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}: loss = {loss.detach().cpu().item()}')
        val_auc, val_ap = model.test(z, val_data.pos_edge_label_index, val_data.neg_edge_label_index)
        print(f'Validation AUC: {val_auc}, validation AP: {val_ap}')
        test_auc, test_ap = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
        print(f'Test AUC: {test_auc}, test AP: {test_ap}')
    

if __name__ == '__main__':
    main()

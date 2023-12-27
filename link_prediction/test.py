import torch
import torch.nn as nn
from torch.optim import Adam
from models import VGAE
import torch_geometric
from torch_geometric import datasets
from sklearn.metrics import roc_auc_score, average_precision_score

def train_val_test_split(data, val=.05, test=.1):
    edge_index = data.edge_index
    num_edges = edge_index.shape[1]
    perm = torch.randperm(num_edges)
    num_val = int(val * num_edges)
    num_test = int(test * num_edges)
    val_edges = edge_index[:,perm[:num_val]]
    test_edges = edge_index[:,perm[num_val:(num_val+num_test)]]
    train_edges = edge_index[:,perm[num_val+num_test:]]
    train_data = torch_geometric.data.Data(x=data.x, edge_index=train_edges, y=data.y)
    return train_data, val_edges, test_edges

def sample_negative_edges(data, val_or_test_edges):
    num_nodes = len(data.x)
    edge_list = data.edge_index.transpose(0, 1)
    edge_set = set([tuple(edge.numpy()) for edge in edge_list])
    sampled_set = set()
    for _ in range(len(val_or_test_edges)):
        while True:
            edge = torch.randint(0, num_nodes, size=(2,))
            edge_tuple = tuple(edge.numpy())
            if edge_tuple not in sampled_set and edge_tuple not in edge_set:
                sampled_set.add(edge_tuple)
                break
    return torch.tensor(list(sampled_set)).transpose(0, 1)

def log_likelihood(adj_matrix, model_output):
    log_sigmoid = -torch.log(1 + torch.exp(-model_output))
    log_probs = torch.where(adj_matrix != 0.0, log_sigmoid, log_sigmoid - model_output)
    return torch.sum(log_probs)

def kl_divergence(mus, sigma2s):
    N, k = mus.shape
    return .5*(k * torch.sum(sigma2s) + torch.tensordot(mus, mus) - N*k - k*torch.sum(torch.log(sigma2s)))

def elbo_estimate(model_output, mus, sigma2s, adj_matrix):
    return log_likelihood(adj_matrix, model_output) - kl_divergence(mus, sigma2s)

def labels_probs(model_output, pos_edge_list, neg_edge_list):
    prob_matrix = torch.sigmoid(model_output).cpu().detach()
    labels = [1] * len(pos_edge_list) + [0] * len(neg_edge_list)
    probs = [prob_matrix[tuple(pos_edge_list[i])] for i in range(len(pos_edge_list))] \
        + [prob_matrix[tuple(neg_edge_list[i])] for i in range(len(neg_edge_list))]
    return labels, probs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = datasets.Planetoid(root='__data/Cora', name='Cora', split='public')
    x_dim = dataset[0].x.shape[1]
    train_data, val_edges, test_edges = train_val_test_split(dataset[0])
    neg_val = sample_negative_edges(dataset[0], val_edges.transpose(0, 1))
    neg_test = sample_negative_edges(dataset[0], test_edges.transpose(0, 1))
    adj_matrix = torch_geometric.utils.to_dense_adj(train_data.edge_index).to(device)
    model = VGAE(x_dim, latent_dim=16, dropout=0.0).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    num_epochs = 200
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, mus, sigma2s = model(train_data.x.to(device), train_data.edge_index.to(device))
        minus_elbo_est = -elbo_estimate(output, mus, sigma2s, adj_matrix)
        minus_elbo_est.backward()
        optimizer.step()
        print(f'Epoch {epoch}: ELBO estimate = {-minus_elbo_est.detach().cpu().item()}')
        labels_val, probs_val = labels_probs(output, val_edges.transpose(0, 1), neg_val.transpose(0, 1))
        labels_test, probs_test = labels_probs(output, test_edges.transpose(0, 1), neg_test.transpose(0, 1))
        val_auc = roc_auc_score(labels_val, probs_val)
        val_ap = average_precision_score(labels_val, probs_val)
        test_auc = roc_auc_score(labels_test, probs_test)
        test_ap = average_precision_score(labels_test, probs_test)
        print(f'Validation AUC: {val_auc}, validation AP: {val_ap}')
        print(f'Test AUC: {test_auc}, test AP: {test_ap}')
    

if __name__ == '__main__':
    main()

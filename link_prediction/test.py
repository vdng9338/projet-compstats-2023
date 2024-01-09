import torch
import torch.nn as nn
from torch.optim import Adam
from models import VGAE
import torch_geometric
from torch_geometric import datasets
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import os

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

def log_likelihood(adj_matrix, model_output, positive_weight):
    log_sigmoid = -torch.log(1 + torch.exp(-model_output))
    log_probs = torch.where(adj_matrix > 0.1, positive_weight*log_sigmoid, log_sigmoid - model_output)
    return torch.sum(log_probs)

def log_likelihood_negative_sampling(model_output, pos_edge_index, neg_edge_index, positive_weight=1.0):
    log_sigmoid = -torch.log(1+torch.exp(-model_output))
    ret = positive_weight*log_sigmoid[tuple(pos_edge_index)].sum()
    ret += (log_sigmoid - model_output)[tuple(neg_edge_index)].sum()
    return ret

def kl_divergence(mus, logsigma2s):
    N, k = mus.shape
    return .5*(torch.sum(torch.exp(logsigma2s)) + torch.tensordot(mus, mus) - N*k - torch.sum(logsigma2s))

def elbo_estimate(model_output, mus, logsigma2s, adj_matrix):
    return log_likelihood(adj_matrix, model_output, 200) - kl_divergence(mus, logsigma2s)

def elbo_estimate_neg_sampling(model_output, mus, logsigma2s, pos_edge_index, neg_edge_index):
    return log_likelihood_negative_sampling(model_output, pos_edge_index, neg_edge_index) - .005*kl_divergence(mus, logsigma2s)

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
    """train_data, val_edges, test_edges = train_val_test_split(dataset[0])
    neg_val = sample_negative_edges(dataset[0], val_edges.transpose(0, 1))
    neg_test = sample_negative_edges(dataset[0], test_edges.transpose(0, 1))"""
    transform = RandomLinkSplit(num_val=.05, num_test=.1, is_undirected=True, split_labels=True)
    train_data, val_data, test_data = transform(dataset[0])
    adj_matrix = torch_geometric.utils.to_dense_adj(train_data.edge_index).to(device)
    model = VGAE(x_dim, dropout=0.0).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    num_epochs = 200
    elbos = np.zeros(num_epochs)
    train_aucs = np.zeros(num_epochs)
    train_aps = np.zeros(num_epochs)
    val_aucs = np.zeros(num_epochs)
    val_aps = np.zeros(num_epochs)
    test_aucs = np.zeros(num_epochs)
    test_aps = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, mus, logsigma2s = model(train_data.x.to(device), train_data.edge_index.to(device))
        train_neg_edge_index = negative_sampling(train_data.edge_index, force_undirected=True)
        #minus_elbo_est = -elbo_estimate(output, mus, logsigma2s, adj_matrix)
        minus_elbo_est = -elbo_estimate_neg_sampling(output, mus, logsigma2s, train_data.edge_index, train_neg_edge_index)
        minus_elbo_est.backward()
        optimizer.step()
        elbo = -minus_elbo_est.detach().cpu().item()
        print(f'Epoch {epoch}: ELBO estimate = {elbo}')
        elbos[epoch] = elbo
        labels_train, probs_train = labels_probs(output, train_data.pos_edge_label_index.transpose(0, 1), train_data.neg_edge_label_index.transpose(0, 1))
        labels_val, probs_val = labels_probs(output, val_data.pos_edge_label_index.transpose(0, 1), val_data.neg_edge_label_index.transpose(0, 1))
        labels_test, probs_test = labels_probs(output, test_data.pos_edge_label_index.transpose(0, 1), test_data.neg_edge_label_index.transpose(0, 1))
        train_auc = roc_auc_score(labels_train, probs_train)
        train_ap = average_precision_score(labels_train, probs_train)
        val_auc = roc_auc_score(labels_val, probs_val)
        val_ap = average_precision_score(labels_val, probs_val)
        test_auc = roc_auc_score(labels_test, probs_test)
        test_ap = average_precision_score(labels_test, probs_test)
        print(f'Train AUC: {train_auc}, train AP: {train_ap}')
        print(f'Validation AUC: {val_auc}, validation AP: {val_ap}')
        print(f'Test AUC: {test_auc}, test AP: {test_ap}')
        train_aucs[epoch] = train_auc
        train_aps[epoch] = train_ap
        val_aucs[epoch] = val_auc
        val_aps[epoch] = val_ap
        test_aucs[epoch] = test_auc
        test_aps[epoch] = test_ap
    path_to_figures = "Slides/figures"
    fig = plt.figure()
    plt.plot(elbos)
    plt.title("ELBO estimates per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO estimate")
    plt.show()
    fig.savefig(os.path.join(path_to_figures, "normal_elbo_estimates.pdf"))
    fig = plt.figure()
    plt.plot(train_aucs, label="Train")
    plt.plot(val_aucs, label="Validation")
    plt.plot(test_aucs, label="Test")
    plt.title("ROC AUC per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.legend(loc="best")
    plt.show()
    fig.savefig(os.path.join(path_to_figures, "normal_auc.pdf"))
    fig = plt.figure()
    plt.plot(train_aps, label="Train")
    plt.plot(val_aps, label="Validation")
    plt.plot(test_aps, label="Test")
    plt.title("Average precision per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average precision")
    plt.legend(loc="best")
    plt.show()
    fig.savefig(os.path.join(path_to_figures, "normal_ap.pdf"))

if __name__ == '__main__':
    main()

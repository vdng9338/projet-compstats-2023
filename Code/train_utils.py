import torch
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score

def log_likelihood(adj_matrix, model_output, positive_weight):
    log_sigmoid = -torch.log(1 + torch.exp(-model_output))
    log_probs = torch.where(adj_matrix > 0.1, positive_weight*log_sigmoid, log_sigmoid - model_output)
    return torch.sum(log_probs)

# def kl_divergence_vMF():
#     """ KL divergence between vMF and uniform distribution 
#         in the hypersphere
#     """

# def train(model):

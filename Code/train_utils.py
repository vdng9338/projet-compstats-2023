import torch
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score



def get_edge_probs(model_output : torch.Tensor,
                 pos_edge_index : torch.Tensor,
                 neg_edge_index : torch.Tensor):
    """ corresponds to labels_probs function from Victor's code """
    nb_pos_edges = pos_edge_index.shape[-1]
    nb_neg_edges = neg_edge_index.shape[-1]
    # turn output into probs
    out_probs = torch.sigmoid(model_output).cpu().detach()
    # get labels
    labels = [1] * nb_pos_edges + [0] * nb_neg_edges
    # edge_probs = [out_probs[tuple(pos_edge_index[:, i])].item() for i in range(nb_pos_edges)] + []
    edge_probs = out_probs[tuple(pos_edge_index)].tolist() + out_probs[tuple(neg_edge_index)].tolist()
    return labels, edge_probs

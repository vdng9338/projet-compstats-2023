import torch
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score



def get_edge_probs(model_output : torch.Tensor,
                 edge_label_index : torch.Tensor):
    """ corresponds to labels_probs function from Victor's code """
    nb_edges = edge_label_index.shape[-1]
    # turn output into probs
    out_probs = torch.sigmoid(model_output).cpu().detach()
    # get labels
    edge_probs = [out_probs[tuple(edge_label_index[:, i])].item() for i in range(nb_edges)]
    
    return edge_probs

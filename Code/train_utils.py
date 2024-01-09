import torch
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib.pyplot as plt


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


def plot_curves(metrics: dict, figure_path : str):
    fig, ax = plt.subplots(1, 3)

    ax[0].plot(metrics['train_loss'])
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Train loss')

    ax[1].plot(metrics['train_auc'], label='train')
    ax[1].plot(metrics['val_auc'], label='val')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('AUC')
    ax[1].legend()
    ax[1].set_title('Area under the curve during training')

    ax[1].plot(metrics['train_auc'], label='train')
    ax[1].plot(metrics['val_auc'], label='val')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('AUC')
    ax[1].legend()
    ax[1].set_title('Area under the curve during training')

    ax[2].plot(metrics['train_ap'], label='train')
    ax[2].plot(metrics['val_ap'], label='val')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('ap')
    ax[2].legend()
    ax[2].set_title('Average precision during training')

    # plt.show() 
    fig.savefig(figure_path) 
    

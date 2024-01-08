from typing import Any

import torch 
from scipy.special import iv, ive 

def reconstruction_loss(model_output: torch.Tensor,
                   edge_label_index: torch.Tensor,
                   edge_label:torch.Tensor, 
                   positive_weight: float = 1):
    
    log_sigmoid = -torch.log(1 + torch.exp(-model_output))
    log_probs = log_sigmoid[tuple(edge_label_index)]
    out_edge = model_output[tuple(edge_label_index)]
    log_lik = torch.where(edge_label ==1, positive_weight*log_probs, log_probs - out_edge)
    return log_lik.mean() # sum or mean ?

# class reconstruction_loss(torch.autograd.Function):
#     """ Reconstruction loss for vMF-VAE"""

#     @staticmethod
#     def forward(ctx, x, ind):
#         pass

#     @staticmethod
#     def backward(ctx, grad_x, grad_ind):
#         pass

class kl_div_vmf(torch.autograd.Function):
    """ KL divergence between vMF and uniform distribution in the hypersphere"""

    @staticmethod
    def forward(ctx, mus, kappas):
        """"
        ctx : context object to save tensors for backward pass
        """
        dim = mus.shape[-1]
        bessel_1 = iv(dim/2, kappas) # I_{d/2}
        bessel_2 = iv(dim/2 - 1, kappas)

        ctx.save_for_backward(torch.tensor(dim), kappas)
        kl = kappas * bessel_1/bessel_2 + ( (dim/2 - 1)*torch.log(kappas) - (dim/2)*torch.log(torch.tensor(2*torch.pi)) - torch.log(bessel_2) ) \
            + (dim/2)*torch.log(torch.tensor(torch.pi)) + torch.log(torch.tensor(2)) - torch.lgamma(torch.tensor(dim/2))
        return kl.mean() # sum or mean ?
        
    @staticmethod
    def backward(ctx, grad_kappa):

        dim, kappas = ctx.saved_tensors
        dim = dim.item()
        g = .5 * kappas * ( ive(dim/2 + 1, kappas) / ive(dim/2 - 1, kappas) - ive(dim/2, kappas) * ( ive(dim/2 - 2, kappas) + ive(dim/2, kappas) ) / ive(dim/2 - 1, kappas)**2 + 1)

        return None, g * grad_kappa
    

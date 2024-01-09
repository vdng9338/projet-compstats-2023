from typing import Any

import torch 
from scipy.special import iv, ive
import numpy as np

class Log_VMF_normalizing_constant(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, kappa, m):
        ctx.save_for_backward(kappa)
        ctx.dim_m = m
        bessel_value = torch.tensor(iv(m/2 - 1, kappa.cpu().detach().numpy()))

        if torch.any(bessel_value == 0.0) :
            print(f"Warning: some Bessel value is 0. min(kappa) = {torch.min(kappa)}, m={m}")
        return torch.where(bessel_value == 0.0, 0.0, (m/2-1)*torch.log(kappa) - m/2*np.log(2*np.pi) - torch.log(bessel_value))
    
    @staticmethod
    def backward(ctx, grad_output):
        kappa, = ctx.saved_tensors
        kappa = kappa.cpu().detach().numpy()
        m = ctx.dim_m
        return grad_output * torch.tensor(-ive(m/2, kappa)/(ive(m/2-1, kappa)+1e-6)), None

class ExpectedReconstrLoss(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, log_likelihood, inside_g_cor):
        ctx.save_for_backward(log_likelihood)
        return log_likelihood
    
    @staticmethod
    def backward(ctx, grad_output):
        log_likelihood, = ctx.saved_tensors
        return grad_output, torch.exp(log_likelihood)*grad_output


def iv_torch(order, x):
    """ to handle cases where device is not cpu """
    return iv(order, x.cpu()).to(x.device)

def ive_torch(order, x):
    return ive(order, x.cpu()).to(x.device)

def reconstruction_loss(model_output: torch.Tensor,
                   pos_edge_index: torch.Tensor,
                   neg_edge_index:torch.Tensor,
                   ws: torch.Tensor,
                   kappas: torch.Tensor,
                   bs: torch.Tensor,
                   epss: torch.Tensor,
                   m: int,
                   positive_weight: float = 1):
    
    log_sigmoid = -torch.log(1 + torch.exp(-model_output))
    neg_value_probs = log_sigmoid[tuple(neg_edge_index)] - model_output[tuple(neg_edge_index)]
    
    log_likelihood = positive_weight*log_sigmoid[tuple(pos_edge_index)].sum() 

    log_likelihood += neg_value_probs.sum()

    inside_g_cor = torch.sum(
        Log_VMF_normalizing_constant.apply(kappas, m) + ws*kappas
        + .5*(m-3)*torch.log(1-ws**2) + torch.log(torch.abs(-2*bs/((bs-1)*epss+1)**2)))

    return -ExpectedReconstrLoss.apply(log_likelihood, inside_g_cor) # sum or mean ?

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
    def forward(ctx, kappas, mus):
        """"
        ctx : context object to save tensors for backward pass
        """
        dim = mus.shape[-1]
        bessel_1 = iv_torch(dim/2, kappas) # I_{d/2}(\kappa)
        bessel_2 = iv_torch(dim/2 - 1, kappas)


        ctx.save_for_backward(torch.tensor(dim), kappas)
        kl = kappas * bessel_1/(bessel_2+1e-6) + ( (dim/2 - 1)*torch.log(kappas) - (dim/2)*torch.log(torch.tensor(2*torch.pi)) - torch.log(bessel_2) ) \
            + (dim/2)*torch.log(torch.tensor(torch.pi)) + torch.log(torch.tensor(2)) - torch.lgamma(torch.tensor(dim/2))
        return kl.sum()
        
    @staticmethod
    def backward(ctx, grad_kappa):

        dim, kappas = ctx.saved_tensors
        dim = dim.item()
        g = .5 * kappas * ( ive_torch(dim/2 + 1, kappas) / (ive_torch(dim/2 - 1, kappas) + 1e-6) - ive_torch(dim/2, kappas) * ( ive_torch(dim/2 - 2, kappas) + ive_torch(dim/2, kappas) ) / (ive_torch(dim/2 - 1, kappas)+1e-6)**2 + 1)
        # print('grad_kappa', grad_kappa.shape) # here, grad_kappa = []
        # print('g', g.shape)
        return g, None
    

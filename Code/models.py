import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform, Normal
import numpy as np
from torch_geometric.nn.conv import GCNConv
from typing import Literal
from vMF_distribution import VonMisesFisher

from von_mises_fisher import VonMisesFisherClone

class VGAEEncoder(nn.Module):
    
    def __init__(
            self,
            input_dim: int,
            interm_dim: int = 32,
            latent_dim: int = 64,
            latent_distr: Literal['vMF', 'normal'] = 'normal',
            dropout: float = .1
    ):
        super().__init__()
        self.conv1 = GCNConv(input_dim, interm_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2_mu = GCNConv(interm_dim, latent_dim)
        if latent_distr == 'normal':
            self.conv2_var = GCNConv(interm_dim, latent_dim)
        else:
            self.conv2_var = GCNConv(interm_dim, 1)
        # self.conv2_sigma2_kappa = GCNConv(interm_dim, 1)
        self.softplus = nn.Softplus()
        self.latent_distr = latent_distr

    def forward(self, X, graph):
        X = self.conv1(X, graph)
        X = self.relu1(X)
        X = self.dropout1(X)
        mus = self.conv2_mu(X, graph)
        logsigma2s_kappas = self.conv2_var(X, graph)

        if self.latent_distr == 'vMF':
            mus = F.normalize(mus, dim=-1)
            logsigma2s_kappas = logsigma2s_kappas.squeeze(-1)
        return mus, logsigma2s_kappas
    
class VGAE(nn.Module):

    def __init__(
            self,
            input_dim: int,
            interm_dim: int = 32,
            latent_dim: int = 64,
            latent_distr: Literal['vMF', 'normal'] = 'normal',
            dropout: float = .1
    ):
        super().__init__()
        self.encoder = VGAEEncoder(input_dim, interm_dim=interm_dim, latent_dim=latent_dim, latent_distr=latent_distr, dropout=dropout)
        self.latent_distr = latent_distr
        self.latent_dim = latent_dim
        if latent_distr == 'normal':
            # Allows moving the normal distribution to cuda
            # self.normal_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            # self.normal_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            normal_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            normal_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.normal = Normal(normal_mean, normal_std)
            # self.normal = Normal(torch.zeros(latent_dim, requires_grad=False), torch.ones(latent_dim, requires_grad=False))

    
    def forward(self, X, graph):
        """ 
        Parameters:
        graph : is an edge_index tensor (2, nb_edges)
        
        Returns:
        ZZt : output of the model
        mus : 
        logsigma2s :
        logkappas :
        """
        if self.latent_distr == 'vMF':
            mus, logkappas = self.encoder(X, graph)
            kappas = torch.exp(logkappas) # in the github, there is no exp but they add # the `+ 1` prevent collapsing behaviors
            # print('BEF log_kappa', logkappas)
            # print('BEF kappa', kappas)
            vmf = VonMisesFisher(mus, kappas)
            Z, ws, epss, bs = vmf.sample()

            # TRY SAMPLE WITH THE AUTHOR'S CODE
            # q_z = VonMisesFisherClone(mus, kappas)
            # Z = q_z.rsample()

        else:
            mus, logsigma2s = self.encoder(X, graph)
            sigmas = torch.exp(.5*logsigma2s)
            eps = self.normal.sample(mus.size()).to(mus.device)
            Z = mus + sigmas * eps
  
        ZZt = torch.matmul(Z, Z.transpose(0, 1))
        if self.latent_distr == 'vMF':
            return ZZt, mus, logkappas, ws, epss, bs
        else:
            return ZZt, mus, logsigma2s

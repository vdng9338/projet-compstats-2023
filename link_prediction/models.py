import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform, Normal
import numpy as np
from torch_geometric.nn.conv import GCNConv
from typing import Literal

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
        self.conv2_sigma2_kappa = GCNConv(interm_dim, 1)
        self.softplus = nn.Softplus()
        self.latent_distr = latent_distr

    def forward(self, X, graph):
        X = self.conv1(X, graph)
        X = self.relu1(X)
        X = self.dropout1(X)
        mus = self.conv2_mu(X)
        sigma2s_kappas = self.softplus(self.conv2_sigma2_kappa(X))
        if self.latent_distr == 'vMF':
            mus = F.normalize(mus, dim=-1)
        return mus, sigma2s_kappas
    
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
            self.normal_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.normal_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.normal = Normal(self.normal_mean, self.normal_std)
    
    def forward(self, X, graph):
        if self.latent_distr == 'vMF':
            mus, kappas = self.encoder(X, graph)
            m = self.latent_dim
            n = X.size(0)
            b = (-2*kappas + torch.sqrt(4*kappas**2 + (m-1)**2))/(m-1)
            a = (m-1 + 2*kappas + torch.sqrt(4*kappas**2 + (m-1)**2))/4
            d = (4*a*b)/(1+b) - (m-1)*np.log(m-1)
            sampled = torch.zeros(n, dtype=torch.bool, device=X.device)
            raise ValueError # TODO: Implement
        else:
            mus, sigma2s = self.encoder(X, graph)
            sigmas = torch.sqrt(sigma2s)
            eps = self.normal.sample(mus.size())
            Z = mus + sigmas.unsqueeze(1) * eps
        return F.sigmoid(torch.matmul(Z, Z.transpose(0, 1)))

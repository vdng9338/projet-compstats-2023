import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform, Normal, Beta
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
        if latent_distr == "normal":
            self.conv2_sigma2_kappa = GCNConv(interm_dim, latent_dim)
        else:
            self.conv2_sigma2_kappa = GCNConv(interm_dim, 1)
        self.softplus = nn.Softplus()
        self.latent_distr = latent_distr

    def forward(self, X, graph):
        X = self.conv1(X, graph)
        X = self.relu1(X)
        X = self.dropout1(X)
        mus = self.conv2_mu(X, graph)
        logsigma2s_kappas = self.conv2_sigma2_kappa(X, graph)
        if self.latent_distr == 'vMF':
            logsigma2s_kappas = torch.squeeze(logsigma2s_kappas, -1)
            mus = F.normalize(mus, dim=-1)
            logsigma2s_kappas = self.softplus(logsigma2s_kappas)
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
        if self.latent_distr == 'vMF':
            self.beta_conc1 = nn.Parameter(torch.tensor((self.latent_dim-1)/2), requires_grad=False)
            self.beta_conc0 = nn.Parameter(torch.tensor((self.latent_dim-1)/2), requires_grad=False)
            self.beta_distr = Beta(self.beta_conc1, self.beta_conc0)

    def sample_w3D(self, kappas):
        """ Faster method in the 3D case (inverse method + exponential function identities).
        This method is much faster than the general rejection sampling based algorithm.
          """
        assert self.latent_dim == 3
        n = kappas.size(0)

        u = torch.rand(n, device=kappas.device)
        w = 1 + torch.log(u + (1 - u) * torch.exp(-2 * kappas)) / kappas
        return w

    def sample_w(self, kappas):
        """ acceptance rejection sampling """
        m = self.latent_dim
        n = kappas.size(0)
        sqrt = torch.sqrt(4 * kappas**2 + (m-1)**2)
        b = (-2*kappas + sqrt) / (m-1)
        a = ((m-1) + 2*kappas + sqrt) / 4
        d = 4*a*b / (1+b) - (m -1) * np.log(m-1)
        w = torch.zeros(n, device=kappas.device)
        w_sampled = torch.zeros(n, device=kappas.device, dtype=torch.bool)
        while ~torch.all(w_sampled):
            eps = self.beta_distr.sample((n,))
            w_prop = (1 - (1+b)*eps) / (1 - (1-b)*eps)
            t = 2*a*b / (1 - (1-b)*eps)

            u = torch.rand(n, device=kappas.device)
            cond = (m -1)*torch.log(t) - t + d

            w = torch.where(~w_sampled, w_prop, w)
            w_sampled |= (cond >= torch.log(u))
        return w 
    
    def householder_transform(self, x, mus):
        e = torch.zeros((1, self.latent_dim), device=x.device)
        e[0, 0] = 1
        u = e - mus
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)

        return x - 2 * (x*u).sum(-1, keepdim=True) * u
    
    def sample_vMF(self, mus, kappas):
        n = mus.size(0)
        m = self.latent_dim
        if m == 3:
            w = self.sample_w3D(kappas)[..., None]
        else : # _TODO : case 2D
            w = self.sample_w(kappas)[..., None]

        v = torch.randn((n, m-1), device=mus.device)
        v = v / torch.norm(v, dim=1)[..., None]

        z = torch.cat([w, torch.sqrt(1 - w**2)*v], axis=-1)
        z = self.householder_transform(z, mus)
        return z
    
    def forward(self, X, graph):
        if self.latent_distr == 'vMF':
            mus, kappas = self.encoder(X, graph)
            #kappas = torch.exp(logkappas)
            Z = self.sample_vMF(mus, kappas)
        else:
            mus, logsigma2s = self.encoder(X, graph)
            sigmas = torch.exp(.5*logsigma2s)
            eps = torch.randn(mus.size(), device=mus.device)
            Z = mus + sigmas * eps
        ZZt = torch.matmul(Z, Z.transpose(0, 1))
        if self.latent_distr == 'vMF':
            return ZZt, mus, kappas
        else:
            return ZZt, mus, logsigma2s

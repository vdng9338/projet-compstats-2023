import numpy as np
import torch
from typing import Optional, Union
from scipy.special import iv # modified Bessel function of the first kind of real order and complex argument.
from scipy.stats import beta 
# from torch.distributions import multivariate_normal

class VonMisesFisher():
    def __init__(self, mus : torch.Tensor, kappas : Union[int, torch.Tensor]):
        self.mus = mus
        self.dtype = mus.dtype
        if isinstance(kappas, torch.Tensor):
            self.kappas = kappas
        else:
            self.kappas = torch.tensor([kappas], dtype=self.dtype)
        self.dim = mus.shape[-1]

        self._cst = None
        

    def compute_cst(self):
        self._cst = []

        for i in range(self.kappas.shape[0]):
            bessel_value = iv(self.dim/2 - 1, self.kappas[i])

            if bessel_value == 0 :
                print("Warning: Bessel value is 0")
                self._cst.append(1)
            else :
                self._cst.append(self.kappas[i]**(self.dim/2 - 1)/( (2*np.pi)**(self.dim/2) * bessel_value ))

    def pdf(self, x : torch.Tensor):
        """ x is a random unit vector in R^d"""
        assert len(x) == self.dim
        assert np.isclose(np.linalg.norm(x), 1), f'x {x} is not a unit vector: {np.linalg.norm(x)}'
        if not self._cst:
            self.compute_cst()
        values = []
        for i in range(self.kappas.shape[0]):
            dot = torch.einsum('i, ...i -> ...', self.mus[i], x)
            log_val = self.kappas[i] * dot # self.mu.dot(x)
            values.append(self._cst[i]*np.exp(log_val))

        return torch.tensor(values, dtype=self.dtype)
    
    def sample_w3D(self, kappa : float, n_samples):
        """ more faster method in the 3D case (inverse method + exponential function identities).
        This method is much faster than the general rejection sampling based algorithm.
          """
        assert self.dim == 3

        u = torch.rand(n_samples)
        w = 1 + torch.log(u + (1 - u) * torch.exp(-2 * kappa)) / kappa
        return w
    
    def sample_w(self, kappa : float, n_samples):
        """ acceptance rejection sampling """
        sqrt = torch.sqrt(4 * kappa**2 + (self.dim-1)**2)
        b = (-2*kappa + sqrt) / (self.dim-1)

        a = ((self.dim-1) + 2*kappa + sqrt) / 4

        d = 4*a*b / (1+b) - (self.dim -1) * np.log(self.dim-1)
        w = []
        count = 0
        while len(w) < n_samples :
            eps = beta.rvs((self.dim -1)/2, (self.dim -1)/2, size=1)
            w_prop = (1 - (1+b)*eps[0]) / (1 - (1-b)*eps[0])
            t = 2*a*b / (1 - (1-b)*eps[0])
            u = np.random.uniform()
            cond = (self.dim - 1)*torch.log(t) - t + d
            if cond >= np.log(u):
                w.append(w_prop)
                continue

            count += 1
            if count > 1e6:
                print('Warning : the while loop is too long')
                return torch.tensor(w, dtype=self.dtype)
            
        return torch.tensor(w, dtype=self.dtype) 

    def householder_transform(self, mu, x):
        e = torch.zeros(self.dim)
        e[0] = 1
        u = e - mu
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)

        return x - 2 * (x*u).sum(-1, keepdim=True) * u
    
    def sample(self, n_samples: int = 1):
        z_samples = []
        for i in range(self.kappas.shape[0]):
            if self.dim == 3:
                w = self.sample_w3D(kappa = self.kappas[i], n_samples=n_samples)[..., None]
            else : # _TODO : case 2D
                w = self.sample_w(kappa = self.kappas[i], n_samples=n_samples)[..., None]

            v = np.random.multivariate_normal(mean=np.zeros(self.dim-1), cov=np.eye(self.dim-1), size=n_samples)
            v = v / np.linalg.norm(v, axis=1)[..., None]

            z = torch.squeeze(torch.cat([w, torch.sqrt(1 - w**2)*v], axis=-1))
     
            z_samples.append(self.householder_transform(self.mus[i], z))

        return torch.stack(z_samples)

    
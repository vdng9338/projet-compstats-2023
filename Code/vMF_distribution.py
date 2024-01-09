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
    
    def sample_w(self, kappa : torch.Tensor, n_samples):
        """ acceptance rejection sampling """
        sqrt = torch.sqrt(4 * kappa**2 + (self.dim-1)**2)
        b = (-2*kappa + sqrt) / (self.dim-1)

        a = ((self.dim-1) + 2*kappa + sqrt) / 4

        d = 4*a*b / (1+b) - (self.dim -1) * np.log(self.dim-1)
        w = torch.zeros(n_samples, device=kappa.device)
        epss = torch.zeros(n_samples, device=kappa.device)
        for i in range(n_samples):
            count = 0
            while True:
                eps = torch.tensor(beta.rvs((self.dim -1)/2, (self.dim -1)/2, size=1), device=kappa.device)
                w_prop = (1 - (1+b)*eps[0]) / (1 - (1-b)*eps[0])
                t = 2*a*b / (1 - (1-b)*eps[0])
                u = torch.rand(())
                cond = (self.dim - 1)*torch.log(t) - t + d
                if cond >= torch.log(u):
                    w[i] = w_prop
                    epss[i] = eps
                    break

                count += 1
                if count > 1e7:
                    print('Warning : the while loop is too long')
                    return w, epss, b
        return w, epss, b 

    def householder_transform(self, mu, x):
        e = torch.zeros(self.dim, device=mu.device)
        e[0] = 1
        u = e - mu
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)

        return x - 2 * (x*u).sum(-1, keepdim=True) * u
    
    def sample(self, n_samples: int = 1):
        """ 
        Return :
        z_samples : samples of size (n_mus, n_samples, self.dim) 
            (n_mus is the number of set of parameters theta = (mu, kappa))
            if n_samples == 1, z_samples is of size (n_mus, self.dim)
        """
        z_samples = []
        wss = []
        epsss = []
        bs = []
        for i in range(self.kappas.shape[0]):
            if self.dim == 3:
                w = self.sample_w3D(kappa = self.kappas[i], n_samples=n_samples)[..., None] # (n_samples, 1)

            else : # _TODO : case 2D
                w, epss, b = self.sample_w(kappa = self.kappas[i], n_samples=n_samples)
                w = w[..., None] # (n_samples, 1)

            #v = np.random.multivariate_normal(mean=np.zeros(self.dim-1), cov=np.eye(self.dim-1), size=n_samples)
            #v = v / np.linalg.norm(v, axis=1)[..., None]
            v = torch.randn((n_samples, self.dim-1), device=self.kappas.device)
            v = v / torch.norm(v, dim=-1)[..., None] # (n_samples, self.dim-1)


            # w : (n_samples, 1)
            # v : (n_samples, self.dim-1)
            # torch.cat([w, v], axis=-1) : (n_samples, self.dim)
            z = torch.squeeze(torch.cat([w, torch.sqrt(1 - w**2)*v], axis=-1))
 
            if n_samples == 1:
                z = z[0] # (self.dim)
                if self.dim != 3:
                    epss = epss[0]
                    w = w[0]
            z_samples.append(self.householder_transform(self.mus[i], z))
            if self.dim != 3:
                epsss.append(epss)
                wss.append(w.squeeze(-1))
                bs.append(b)

        if self.dim != 3:
            return torch.stack(z_samples), torch.stack(wss), torch.stack(epsss), torch.stack(bs)
        else:
            return torch.stack(z_samples), None, None, None

    
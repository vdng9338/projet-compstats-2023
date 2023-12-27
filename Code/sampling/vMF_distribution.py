import numpy as np
import torch
from scipy.special import iv # modified Bessel function of the first kind of real order and complex argument.
from scipy.stats import beta 
# from torch.distributions import multivariate_normal

class VonMisesFisher():
    def __init__(self, mu, kappa):
        self.mu = mu
        self.dtype = mu.dtype
        self.kappa = torch.tensor(kappa, dtype=self.dtype)
        self.dim = len(mu)
        bessel_value = iv(self.dim/2 - 1, self.kappa)
        if bessel_value == 0 :
            print("Warning: Bessel value is 0")
            self._cst = 1
        else :
            self._cst = kappa**(self.dim/2 - 1)/( (2*np.pi)**(self.dim/2) * bessel_value )


    def pdf(self, x):
        """ x is a random unit vector in R^d"""
        assert len(x) == self.dim
        assert np.isclose(np.linalg.norm(x), 1), f'x {x} is not a unit vector: {np.linalg.norm(x)}'
        dot = np.einsum('i, ...i -> ...', self.mu, x)
        log_val = self.kappa * dot # self.mu.dot(x)
        return   self._cst * np.exp(log_val)
    
    def sample_w3D(self, n_samples):
        """ more faster method in the 3D case (inverse method + exponential function identities).
        This method is much faster than the general rejection sampling based algorithm.
          """
        assert self.dim == 3

        u = torch.rand(n_samples)
        w = 1 + torch.log(u + (1 - u) * torch.exp(-2 * self.kappa)) / self.kappa
        return w
    
    def sample_w(self, n_samples):
        """ acceptance rejection sampling """
        sqrt = np.sqrt(2 * self.kappa**2 + (self.dim-1)**2)
        b = (-2*self.kappa + sqrt) / (self.dim-1)
        a = ((self.dim-1) + 2*self.kappa + sqrt) / 4
        d = 4*a*b / (1+b) - (self.dim -1) * np.log(self.dim-1)
        w = []
        count = 0
        while len(w) < n_samples :
            eps = beta.rvs((self.dim -1)/2, (self.dim -1)/2, size=1)
            w_prop = (1 - (1+b)*eps) / (1 - (1-b)*eps)
            t = 2*a*b / (1 - (1-b)*eps)

            u = np.random.uniform()
            cond = (self.dim -1)*np.log(t) - t + d

            if cond >= u:
                w.append(w_prop)
                continue

            count += 1
            if count > 1e99:
                print('Warning : the while loop is too long')
                return torch.tensor(w, dtype=self.dtype)
        return torch.tensor(w, dtype=self.dtype) 

    def householder_transform(self, x):
        e = torch.zeros(self.dim)
        u = e - self.mu
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)

        return x - 2 * (x*u).sum(-1, keepdim=True) * u
    
    def sample(self, n_samples):
        if self.dim == 3:
            w = self.sample_w3D(n_samples=n_samples)[..., None]
        else : # _TODO : case 2D
            w = self.sample_w(n_samples=n_samples)[..., None]

        v = np.random.multivariate_normal(mean=np.zeros(self.dim-1), cov=np.eye(self.dim-1), size=n_samples)
        v = v / np.linalg.norm(v, axis=1)[..., None]

        z = torch.cat([w, torch.sqrt(1 - w**2)*v], axis=-1)
        z = self.householder_transform(z)
        return z

    

if __name__ == "__main__":
    mu = np.array([1, 0, 0])
    kappa = 1
    print(iv(3, 0))
    vMF = VonMisesFisher(mu, kappa)
    z = np.random.randn(3)
    z = z / np.linalg.norm(z)
    print(vMF.pdf(z))
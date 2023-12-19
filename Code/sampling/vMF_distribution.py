import numpy as np
from scipy.special import jv # Bessel function of the first kind of real order and complex argument.


class VonMisesFisher():
    def __init__(self, mu, kappa):
        self.mu = mu
        self.kappa = kappa
        self.dim = len(mu)
        bessel_value = jv(self.dim/2 - 1, self.kappa)
        if bessel_value == 0 :
            print("Warning: Bessel value is 0")
            self._cst = 1
        else :
            self._cst = self.kappa**(self.dim/2 - 1) / ( (2*np.pi)**(self.dim/2)* bessel_value )

    def pdf(self, x ):
        """ x is a random unit vector in R^d"""
        # assert len(x) == self.dim
        # assert np.linalg.norm(x) == 1

        return self._cst *  np.exp(self.kappa * self.mu.dot(x))
    
    def sample(self, n_samples):
        raise NotImplemented

    

if __name__ == "__main__":
    mu = np.array([1, 0, 0])
    kappa = 1
    print(jv(3, 0))
    vMF = VonMisesFisher(mu, kappa)
    z = np.random.randn(3)
    z = z / np.linalg.norm(z)
    print(vMF.pdf(z))
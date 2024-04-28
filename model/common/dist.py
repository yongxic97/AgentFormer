import torch
from torch import distributions as td


class Normal:

    def __init__(self, mu=None, logvar=None, params=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.sigma = torch.exp(0.5 * self.logvar)

    def rsample(self):
        eps = torch.randn_like(self.sigma)
        return self.mu + eps * self.sigma

    def sample(self):
        return self.rsample()

    def kl(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            kl = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        else:
            term1 = (self.mu - p.mu) / (p.sigma + 1e-8)
            term2 = self.sigma / (p.sigma + 1e-8)
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return kl

    def mode(self):
        return self.mu


class Categorical:

    def __init__(self, probs=None, logits=None, temp=0.01):
        super().__init__()
        self.logits = logits
        self.temp = temp
        if probs is not None:
            self.probs = probs
        else:
            assert logits is not None
            self.probs = torch.softmax(logits, dim=-1)
        self.dist = td.OneHotCategorical(self.probs)

    def rsample(self):
        relatex_dist = td.RelaxedOneHotCategorical(self.temp, self.probs)
        return relatex_dist.rsample()

    def sample(self):
        return self.dist.sample()

    def kl(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            p = Categorical(logits=torch.zeros_like(self.probs))
        kl = td.kl_divergence(self.dist, p.dist)
        return kl

    def mode(self):
        argmax = self.probs.argmax(dim=-1)
        one_hot = torch.zeros_like(self.probs)
        one_hot.scatter_(1, argmax.unsqueeze(1), 1)
        return one_hot


class Beta:
    
        def __init__(self, alpha=None, beta=None, params=None):
            super().__init__()
            if params is not None:
                self.alpha, self.beta = torch.chunk(params, chunks=2, dim=-1)
            else:
                assert alpha is not None
                assert beta is not None
                self.alpha = alpha
                self.beta = beta
            self.dist = td.Beta(self.alpha, self.beta)
    
        def rsample(self):
            return self.dist.rsample()
    
        def sample(self):
            return self.dist.sample()
    
        def kl(self, p=None):
            """ compute KL(q||p) """
            if p is None:
                p = Beta(alpha=torch.ones_like(self.alpha), beta=torch.ones_like(self.beta))
            kl = td.kl_divergence(self.dist, p.dist)
            return kl
    
        def mode(self):
            # if self.alpha.all() > 1 and self.beta.all() > 1:
            #     return (self.alpha - 1) / (self.alpha + self.beta - 2)
            # elif self.alpha == 1 and self.beta == 1:
            #     return 0.5
            # elif self.alpha <= 1 and self.beta > 1:
            #     return 0.0
            # elif self.alpha > 1 and self.beta <= 1:
            #     return 1.0
            # else:
            #     return (0.0,1.0)
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
            # return self.alpha / (self.alpha + self.beta)
        def mean(self):
            return self.alpha / (self.alpha + self.beta)
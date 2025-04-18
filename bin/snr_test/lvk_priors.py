import numpy as np
from pathlib import Path
from scipy.stats import cauchy
from scipy.interpolate import interp1d


class LVKPrimaryMassPrior():
    """
    LVK best estimates for the PL+Peak parameters
    Data from Abbott+ (2022) https://arxiv.org/abs/2111.03634
    Please note that Abbott+ does not provide an estimate for the peak width, we obtained it from the data release directly
    """

    def __init__(self, m_min=3., m_max=100.):
        
        self.alpha_pl   = 3.5   #
        self.mu_peak    = 34.   # Msun
        self.sigma_peak = 4.6   # Msun
        self.w          = 0.038 #
        self.m_min      = 3.    # Msun
        self.m_max      = 100.  # Msun

        # Useful quantities
        self.log_w     = np.log(self.w)
        self.log_1mw   = np.log(1.-self.w)
        self.norm_pl   = np.log(self.alpha_pl-1) - (1-self.alpha_pl)*np.log(self.m_min)
        self.norm_peak = -0.5*np.log(2*np.pi) - np.log(self.sigma_peak)

        # LVK interpolant
        LVK_o3 = np.genfromtxt('lvk_log_plpeak.txt')
        self.pl_peak_interpolant = interp1d(LVK_o3[:,0], LVK_o3[:,1], fill_value = 'extrapolate')

    @property
    def m(self):
        return np.linspace(self.m_min, self.m_max, 10_000)
    
    @property
    def pdf_plot(self):
        return self.pdf(self.m)


    @property
    def pdf_max(self):
        if not hasattr(self, '_pdf_max'):
            m = np.linspace(self.m_min, self.m_max, 10_000)
            self._pdf_max = max(self.pdf(m))
        return self._pdf_max
    
    def log_pdf(self, m):
        """
        LVK Power-law + peak model as in Abbott et al (2022) https://arxiv.org/abs/2111.03634
        Data from https://zenodo.org/record/7843926
        """
        return self.pl_peak_interpolant(m)
    
    def pdf(self, m):
        return np.exp(self.log_pdf(m))
    
    def sample(self, num_samples):
        """Implement a rejection sampler for the LVK prior"""
        samples = []
        attempts = 0
        while len(samples) < num_samples:
            print(f'Remaining {num_samples - len(samples)} primary mass samples to draw', end='\r')
            x = np.random.uniform(self.m_min, self.m_max)
            u = np.random.uniform(0, self.pdf_max)
            p = self.pdf(x)
            if u < p:
                samples.append(x)
            attempts += 1
        efficiency = num_samples / attempts * 100
        print(f"Efficienza del campionamento: {efficiency:.2f}%")
        return samples
    

if __name__=='__main__':
    import matplotlib.pyplot as plt
    prior = LVKPrimaryMassPrior()
    num_samples = int(1e5)
    
    M_samples = prior.sample(num_samples)

    plt.figure()
    plt.hist(M_samples, 100, histtype='step', density=True, label='samples');
    plt.yscale('log')
    plt.semilogy(prior.m, prior.pdf_plot, alpha = 0.8, label='density')
    plt.ylim(1e-5, 1)
    plt.legend()
    plt.xlabel('$m_1$ $[M_{\odot}]$')
    plt.ylabel('$\\dfrac{d\mathcal{R}}{dm_1} [Gpc^{-3} yr^{-1}M_{\odot}^{-1}] $')
    plt.show()
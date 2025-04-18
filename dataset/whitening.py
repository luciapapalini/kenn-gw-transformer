import os
import torch

from tensordict import TensorDict

from hyperion.training import ASD_Sampler
from hyperion.core.fft import rfft, irfft, tukey
from hyperion.core.distributions import UniformPrior

from ..config import CONF_DIR
from ..simulations import EinsteinTelescope
from ..simulations.utils import (matched_filter_snr,
                                 optimal_snr,
                                 network_optimal_snr,
                                 rescale_to_network_snr)

asd_file = os.path.join(CONF_DIR, 'ASD_curves', 'ET_MDC_asd.txt')


class WhitenNet:
    """
    Class that performs whitening and adds Gaussian Noise to the data.
    It can exploit GPU acceleration.

    Constructor Args:
    -----------------
        fs (int):       sampling frequency
        duration (int): duration of the signal
        device (str):   device to use ('cpu' or 'cuda')
    """

    def __init__(self, fs=1024, duration=128, device='cpu'):

        #sampling frequency
        self.fs = fs

        #duration of the signal
        self.duration = duration

        self.device = device

        self.ET = EinsteinTelescope()
        
        #load ASDs
        self.ASDs = {}
        for det in self.ET.arms:
            self.ASDs[det] = ASD_Sampler(fs=fs, 
                                         ifo=det, 
                                         device=device,
                                         duration=duration, 
                                         asd_file=asd_file).asd_reference.type(torch.float64)

        self.SNR_prior = {'BBH': UniformPrior(50, 500, device=device),
                          'BNS': UniformPrior(50, 100, device=device),
                          'NSBH': UniformPrior(50, 100, device=device)}
        return
     

    @property
    def kinds(self):
        return ['BNS', 'NSBH', 'BBH']

    @property
    def n(self):
        return self.duration * self.fs
    
    @property
    def window(self):
        if not hasattr(self, '_window'):
            self._window = tukey(self.n, alpha=0.1, device=self.device)
        return self.window
    
    @property
    def PSDs(self):
        if not hasattr(self, '_PSDs'):
            self._PSDs = {det: self.ASDs[det]**2 for det in self.ET.arms}
        return self._PSDs
    
    @property
    def delta_t(self):
        if not hasattr(self, '_delta_t'):
            self._delta_t = torch.as_tensor(1/self.fs).to(self.device)
        return self._delta_t
    
    @property
    def noise_mean(self):
        if not hasattr(self, '_mu'):
            self._mu = torch.zeros(self.duration*self.fs, device=self.device)
        return self._mu
    
    @property
    def noise_std(self):
        """Rescaling factor to have gaussian noise with unit variance."""
        if not hasattr(self, '_noise_std'):
            self._noise_std = 1 / torch.sqrt(2*self.delta_t)
        return self._noise_std

    

    def add_gaussian_noise(self, h):
        """
        Adds gaussian noise to the whitened signal(s).
        To ensure that noise follows a N(0, 1) distribution we divide by the noise standard deviation
        given by 1/sqrt(2*delta_t) where delta_t is the sampling interval.
        
        Args:
        -----
            h (dict of torch.Tensor): whitened signals
        """
        
        for det in h.keys():
            noise = torch.normal(mean=self.noise_mean, std=self.noise_std)
            h[det] += noise  #/self.noise_std
            
        return h
    
    
    def sum_signals(self, h_i):
        """
        Sum the whitened signals.
        
        Note:
        -----
            h_i has the following structure: [KIND -> DETECTOR -> TIME_SERIES]
            where TIME_SERIES has the shape [BATCH_SIZE, MAX_NUM_SIGNALS, DURATION*FS]
            
            Therefore sum is performed over the second axis.
        
        Args:
        -----
            h (TensorDict object): whitened signals
        """

        out_shape = (h_i.batch_size.numel(), self.duration*self.fs)
        
        h_sum = {det: torch.zeros(out_shape, device=self.device) for det in self.ET.arms}
    
        for det in h_sum:
            for kind in self.kinds:
                h_sum[det] += torch.sum(h_i[kind][det], dim=1)

        return TensorDict.from_dict(h_sum)
    
    
    def rank_by_snr(self, h, snr):

        sorted_snr, id = snr.sort(descending=True, stable=True)
        
        id = id.unsqueeze(-1).expand(-1, -1, self.n)

        for det in h.keys():
            h[det] = h[det].gather(1, id)

        return h , sorted_snr


    def __call__(self, h_i, add_noise=True):

        """
        Whiten the input signal and add Gaussian noise.
        Every signal is furthermore ranked by their SNR.

        Args:
        -----
            h_i (TensorDict): individual input signal(s) [KIND -> DETECTOR -> TIME_SERIES]
            add_noise (bool): whether to add Gaussian noise to the whitened signal(s)

        Returns:
        --------
            h_w (TensorDict): individual whitened signal(s)
            h_sum (TensorDict): sum of the whitened signal(s)

        """
        
        SNR = {}

        for kind in h_i.keys():

            hf = {}

            for det in self.ET.arms:
                h = h_i[kind][det] * tukey(self.n, alpha=0.01, device=self.device)

                #compute the frequency domain signal (template)
                hf[det] = rfft(h, n=self.n) 

                #whiten the signal by dividing wrt the ASD
                hf_w = hf[det] / self.ASDs[det]

                #convert back to the time domain
                h_i[kind][det] = irfft(hf_w, n=self.n)  #/self.noise_std
            
            #compute the optimal SNR
            snr = network_optimal_snr(hf, self.PSDs, self.duration) / self.fs
            

            #rank the signals by SNR
            h_i[kind], SNR[kind] = self.rank_by_snr(h_i[kind], snr)
            
        #sum all the signals
        h_sum = self.sum_signals(h_i)

        if add_noise:
            h_sum = self.add_gaussian_noise(h_sum)
        
        return h_i, h_sum, SNR
        

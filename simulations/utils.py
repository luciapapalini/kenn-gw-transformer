"""Definition of useful functions for the simulations"""

import torch
from hyperion.core.fft import rfft

from astropy.cosmology import Planck18

#======================================
# Cosmology
#======================================

def luminosity_distance_from_redshift(z):
    """
    Computes the luminosity distance from the redshift using the Planck18 cosmology.
    
    Args:
    -----
        z (float or torch.Tensor): redshift
    """
    return Planck18.luminosity_distance(z).value


#======================================
# Matched filter  & SNR
#======================================

def noise_weighted_inner_product(a, b, psd, duration):
    """
    Computes the noise weighte inner product of two frequency domain signals a and b.
    
    Args:
    -----
        a (torch.Tensor): frequency domain signal
        b (torch.Tensor): frequency domain signal
        psd (torch.Tensor): power spectral density
        duration (float): duration of the signal
    
    """
    integrand = torch.conj(a) * b / psd
    return (4 / duration) * torch.sum(integrand, dim = -1)


def optimal_snr(frequency_domain_template, psd, duration):
    """
    Computes the optimal SNR of a signal.
    The code is adapted from Bilby 
    (https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/gw/utils.py?ref_type=heads)
    
    Args:
    -----
        frequency_domain_template (torch.Tensor): frequency domain signal
        psd (torch.Tensor): power spectral density
        duration (float): duration of the signal
    """
    rho_opt = noise_weighted_inner_product(frequency_domain_template, 
                                           frequency_domain_template, 
                                           psd, duration)
    
    snr_square = torch.abs(rho_opt)
    return torch.sqrt(snr_square)
    
    
def matched_filter_snr(frequency_domain_template, frequency_domain_strain, psd, duration):
    """
    Computes the matched filter SNR of a signal.
    The code is adapted from Bilby 
    (https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/gw/utils.py?ref_type=heads)
    
    Args:
    -----
        frequency_domain_template (torch.Tensor): frequency domain template signal
        frequency_domain_strain (torch.Tensor): frequency domain signal
        psd (torch.Tensor): power spectral density
        duration (float): duration of the signal
    """
    rho = noise_weighted_inner_product(frequency_domain_template, 
                                       frequency_domain_strain, 
                                       psd, duration)
    
    rho_opt = noise_weighted_inner_product(frequency_domain_template, 
                                           frequency_domain_template, 
                                           psd, duration)
    
    snr_square = torch.abs(rho / torch.sqrt(rho_opt))
    return torch.sqrt(snr_square)


def network_optimal_snr(frequency_domain_strain, asd, duration):
    """
    Computes the network SNR of a signal given by
    
    SNR_net = [sum (snr_i ^2) ] ^(1/2)

    Args:
    -----
        frequency_domain_strain (dict of torch.Tensor): frequency domain signals
        psd (torch.Tensor): power spectral density
        duration (float): duration of the signal
    """
    
    snr = 0
    for det in frequency_domain_strain.keys():
        #NOTE - to obtain the PSD we first double the asd to avoid numerical issues
        psd = asd[det].double()**2
        snr += optimal_snr(frequency_domain_strain[det], psd, duration)**2

        #print(f'Optimal SNR for {det}: {optimal_snr(frequency_domain_strain[det], psd, duration)}')
    #print(f'Network SNR: {torch.sqrt(snr)}')
    return torch.sqrt(snr)
        
    #return torch.sqrt(torch.sum(torch.stack(det_snr)**2, dim = -1))

    

    
def rescale_to_network_snr(h, new_snr, asd, duration, fs, old_snr = None):
    """
    Rescales the input signal to a new network SNR. 
    
    
    Args:
    -----
        h (dict of torch.Tensor):        Time domain signals. 
        old_snr (float or torch.Tensor): old network SNR. If None it will be computed.
        new_snr (float or torch.Tensor): new network SNR
        kwargs:                          additional arguments to pass to the optimal_snr function. 
                                         (e.g. the sampling frequency to compute the fft)
    
    Returns:
    --------
        hnew (dict of torch.Tensor): rescaled time domain signals
    """
    
    h_o = h.copy()
    
    if old_snr is None:
        #compute the fft of the signals
        hf = torch.stack([h[key] for key in h.keys()])      #we stack the various waveforms together
        hf = rfft(hf, n=hf.shape[-1], norm=fs)         #as pytorch is faster with batched ffts
        #NOTE - the right fft normalization is done by the sampling frequency
        hf_dict = {key: hf[i] for i, key in enumerate(h.keys())} #then we reconvert to a dictionary

        #compute the old snr
        old_snr = network_optimal_snr(hf_dict, asd, duration).unsqueeze(-1)
        
        
    for det in h:
        #print(f'factor for {det}: {new_snr/old_snr}')
        h_o[det] *= (new_snr/old_snr)
    
    '''
    print('============AFTER RESCALING=============')
    hf = torch.stack([h_o[key] for key in h.keys()])      #we stack the various waveforms together
    hf = rfft(hf, n=hf.shape[-1], norm=fs)         #as pytorch is faster with batched ffts
    #NOTE - the right fft normalization is done by the sampling frequency
    hf_dict = {key: hf[i] for i, key in enumerate(h.keys())} #then we reconvert to a dictionary

    #compute the old snr
    old_snr = network_optimal_snr(hf_dict, asd, duration).unsqueeze(-1)
    print('=======================================')
    '''
    return h_o




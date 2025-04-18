
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensordict import TensorDict

from deepfastet.config import CONF_DIR
from deepfastet.simulations import WaveformGenerator, EinsteinTelescope
from deepfastet.simulations.utils import network_optimal_snr

from hyperion.core.fft import rfft
from hyperion.simulations import ASD_Sampler
from hyperion.core.distributions import UniformPrior, CosinePrior

from optparse import OptionParser


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#define waveform generator
wvf_generator = WaveformGenerator(source_kind = 'BBH', signal_duration = 128.0)

#define Einstein Telescope
ET = EinsteinTelescope(use_torch=True, device=device)

#define the PSDs
asd_file = os.path.join(CONF_DIR, 'ASD_curves', 'ET_MDC_asd.txt')
ASDs = {}
for det in ET.arms:
    ASDs[det] = ASD_Sampler(det, asd_file, 
                            fs       = int(wvf_generator.fs),
                            duration = int(wvf_generator.duration),
                            device   = device).asd_reference


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-n", "--n_max", default=10_000, help="Number of samples to draw from the priors. (Default: 10,000)")
    parser.add_option("-m", "--m_min", default=100, help="Minimum mass of the primary black hole. (Default: 100 Msun)")
    (options, args) = parser.parse_args()
   
    m_min = int(options.m_min)
    n_max = int(options.n_max)
    
    # Load the data
    dl    = np.loadtxt('luminosity_distance_samples.txt')*1e3
    z     = np.loadtxt('redshift_samples.txt')
    mass1 = np.loadtxt('primary_mass_samples.txt') * (1+z)
    mass2 = np.random.uniform(0.1, 1, len(mass1))*mass1

    #subsample the data & mask
    i = np.random.choice(len(dl), n_max, replace=False)
    dl, z, mass1, mass2 = dl[i], z[i], mass1[i], mass2[i]

    mask = mass1 >= m_min

    plt.figure()
    plt.hist(mass1, 100, histtype='step', density=True, label='$m_1^z$ samples');
    plt.hist(mass2, 100, histtype='step', density=True, label='$m_2^z$ samples');
    plt.yscale('log')
    plt.xlabel('$m^z$ $[M_{\odot}]$')
    plt.ylabel('$\\dfrac{d\mathcal{R}}{dm_1} [Gpc^{-3} yr^{-1}M_{\odot}^{-1}] $')
    plt.legend()
    plt.savefig('redshifted_masses.png', dpi=200, bbox_inches='tight')
    plt.show()

    #generate the waveforms
    prior_samples = TensorDict.from_dict({'mass1': torch.tensor(mass1[mask]), 'mass2': torch.tensor(mass2[mask]), 'distance': torch.tensor(dl[mask])})
    
    
    print('Generating waveform polarizations...')
    hps, hcs, tcoals = wvf_generator(prior_samples, npool=None)
    
    extrinsic_pars = {}
    extrinsic_pars['ra'] = UniformPrior(0, 2*np.pi).sample((len(prior_samples['mass1']),1))
    extrinsic_pars['dec'] = CosinePrior().sample((len(prior_samples['mass1']),1))
    extrinsic_pars['polarization'] = UniformPrior(0, np.pi).sample((len(prior_samples['mass1']),1))
    
    print('Projecting waveforms into the detector...')
    projected_templates = ET.project_wave(hps, hcs, **extrinsic_pars)
    
    print('Computing network optimal SNR...')
    fd_templates = {det: rfft(projected_templates[det], n=int(wvf_generator.fs*wvf_generator.duration), norm=wvf_generator.fs) for det in projected_templates.keys()}
    network_snr = network_optimal_snr(fd_templates, ASDs, wvf_generator.duration)
    
    
    #save the output
    output = {'mass1': mass1[mask], 'mass2': mass2[mask], 'distance': dl[mask], 'z': z[mask],'network_snr': network_snr.cpu().numpy()}
    
    df = pd.DataFrame.from_dict(output)
    df.to_csv(f'snr_results_m_min_{m_min}.csv', index=False, header=True)
    
    print('Output saved to network_snr.csv')
    
    
    

    
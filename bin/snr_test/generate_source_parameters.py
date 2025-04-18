import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value

from tqdm import tqdm
from lvk_priors import LVKPrimaryMassPrior
from bilby.gw.prior import UniformSourceFrame, UniformComovingVolume

from optparse import OptionParser


if __name__=='__main__':
    parser = OptionParser()
    parser.add_option("-n", "--num_samples", default=100_000, help="Number of samples to draw from the priors. (Default: 100_000)")
    parser.add_option("-z", "--z_max", default=10, help="Maximum redshift to sample the luminosity distance. (Default: 10)")
    (options, args) = parser.parse_args()
    
    num_samples = int(options.num_samples)
    z_max       = float(options.z_max)

    mass_prior = LVKPrimaryMassPrior()
    
    dl_max = Planck18.luminosity_distance(z_max).value
    luminosity_distance_prior = UniformSourceFrame(0, dl_max, cosmology=Planck18, name='luminosity_distance')

    #sample the primary mass 
    M_samples = mass_prior.sample(num_samples)
    
    plt.figure()
    plt.hist(M_samples, 100, histtype='step', density=True, label='samples');
    plt.yscale('log')
    plt.semilogy(mass_prior.m, mass_prior.pdf_plot, alpha = 0.8, label='pdf')
    plt.ylim(1e-5, 1)
    plt.legend()
    plt.xlabel('$m_1$ $[M_{\odot}]$')
    plt.ylabel('$\\dfrac{d\mathcal{R}}{dm_1} [Gpc^{-3} yr^{-1}M_{\odot}^{-1}] $')
    plt.minorticks_on()
    plt.savefig('lvk_primary_mass_draws.png', dpi=200, bbox_inches='tight')
    plt.show()

    #save the samples to a file
    np.savetxt('primary_mass_samples.txt', M_samples)


    #sample the luminosity distance & convert to redshift
    dl_samples = np.array(luminosity_distance_prior.sample(num_samples)/1e3) #convert to Gpc
    print('Converting luminosity distance to redshift')
    z_samples  = np.array([z_at_value(Planck18.luminosity_distance, dl_samples[i]*u.Gpc) for i in tqdm(range(num_samples))])


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(dl_samples, 100, histtype='step', density=True, label='samples');
    plt.xlabel('$d_L$ [Gpc]')
    plt.ylabel('$p(d_L)$')
    plt.minorticks_on()

    plt.subplot(1, 2, 2)
    plt.hist(z_samples, 100, histtype='step', density=True, label='samples');
    plt.xlabel('$z$')
    plt.ylabel('$p(z)$')
    plt.minorticks_on()

    plt.savefig('luminosity_distance_redshift_draws.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    #save the samples to a file
    np.savetxt('luminosity_distance_samples.txt', dl_samples*1e3)
    np.savetxt('redshift_samples.txt', z_samples)

    #generating redshifted masses
    print('Generating redshifted masses')
    m1_redshifted = M_samples*(1+z_samples)

    plt.figure()
    plt.hist(m1_redshifted, 100, histtype='step', density=True, label='samples');
    plt.yscale('log')
    #plt.ylim(1e-5, 1)
    plt.legend()
    plt.xlabel('$m_1 (1+z)$ $[M_{\odot}]$')
    plt.ylabel('$\\dfrac{d\mathcal{R}}{dm_1} [Gpc^{-3} yr^{-1}M_{\odot}^{-1}] $')
    plt.minorticks_on()
    plt.savefig('redshifted_primary_mass_draws.png', dpi=200, bbox_inches='tight')
    plt.show()
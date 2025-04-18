"""Class for data loading and preprocessing"""

import os
import yaml
import h5py
import torch
import numpy as np
import re

from tqdm import tqdm
from pycbc.types import TimeSeries
from torch.utils.data import Dataset

from tensordict import TensorDict

from hyperion.training import ASD_Sampler
from hyperion.core.distributions import MultivariatePrior, prior_dict_
from hyperion.simulations import WhitenNet

from hyperion.core.fft import rfft, irfft, rfftfreq

from torch.distributions import Gamma

from ..config import CONF_DIR
from ..simulations import (EinsteinTelescope, 
                           WaveformGenerator,
                           rescale_to_network_snr, 
                           luminosity_distance_from_redshift)

import matplotlib.pyplot as plt

from hyperion.core.distributions import q_uniform_in_components as q_prior
from hyperion.core.fft import rfft, irfft, rfftfreq

try:
    from hyperion.core.utilities import GWLogger
except: #new hyperion implementation
    from hyperion.core.utilities import HYPERION_Logger as GWLogger

log = GWLogger()

asd_file = os.path.join(CONF_DIR, 'ASD_curves', 'ET_MDC_asd.txt')

class DatasetGenerator(Dataset):
    
    """
    Class to generate training dataset. Can work either offline as well as an online (i.e. on training) generator.
    
    Given a specified prior and a waveform generator it generates as output a tuple
    (parameters, whitened_strain)

    """
    def __init__(self, 
                 prior_filepath  = None,
                 signal_duration = 128,
                 batch_size      = 64,
                 max_signals     = {'BNS' : 1, 
                                    'NSBH': 1,
                                    'BBH' : 2},
                 source_kind           = ['BNS', 'NSBH', 'BBH'],
                 fixed_signals         = False,
                 num_fixed_signals     = 2,
                 device                = 'cpu',
                 mode                  = 'training',
                 inference_parameters = None,
                 add_noise             = True,
                 random_seed           = None,
                 plot_template         = False,
                 plot_signal           = False,
                 npool                 = None,
                 **ET_kwargs
                                ):

        """
        Constructor.

        Args:
        -----
        waveform_generator: object
            GWskysim Waveform generator object istance. (Example: the EffectiveFlyByTemplate generator)

        asd_generators: dict of ASD_sampler objects
            Dictionary with hyperion's ASD_sampler object instances for each interferometer to simulate            

        prior_filepath: str or Path
            Path to a json file specifying the prior distributions over the simulation's parameters

        signal_duration: float
            Duration (seconds) of the output strain. (Default: 1)

        noise_duration : float
            Duration (seconds) of noise to simulate. Setting it higher than duration helps to avoid
            border discontinuity issues when performing whitening. (Default: 2)
        
        batch_size : int
            Batch size dimension. (Default: 64)

        max_signals : dict
            Maximum number of signals to generate for each source kind. (Default: {'BNS': 1, 'NSBH': 1, 'BBH': 2})
        
        source_kind : list of strings
            List of source kinds to generate. (Default: ['BNS', 'NSBH', 'BBH'])

        train_split : float
            Fraction of the dataset to be used for training. (Default: 0.9)
        
        mode : str
            Mode of the generator. Either 'training' or 'validation'. (Default: 'training')

        fixed_signals : bool
            If True, the generator will return a fixed number of superposed signal in each chunk (Default: False)

        num_fixed_signals : int
            Number of fixed signals to superpose in each chunk. (Default: 2)
            
        device : str
            Device to be used to generate the dataset. Either 'cpu' or 'cuda:n'. (Default: 'cpu')

        inference_parameters : list of strings
            List of inference parameter names (e.g. ['m1', 'm2', 'ra', 'dec', ...]). (Default: 
            ['M', 'q', 'e0', 'p_0', 'distance', 'time_shift', 'polarization', 'inclination', 'ra', 'dec'])

        random_seed : int
            Random seed to set the random number generator for reproducibility. (Default: 123)
        
        """

        self.batch_size           = batch_size
        self.signal_duration      = signal_duration
        self.device               = device
        self.mode                 = mode
        self.inference_parameters = inference_parameters
        self.fixed_signals        = fixed_signals
        self.num_fixed_signals    = num_fixed_signals
        self.max_signals          = max_signals
        self.source_kind          = source_kind
        self.add_noise            = add_noise
        self.plot_template        = plot_template
        self.plot_signal_noise    = plot_signal
        self.npool                = npool

        # load Einstein Telescope detector
        self.ET = EinsteinTelescope(device=device, use_torch=True)


        # define mode
        if self.mode == 'training':
            self.num_preload = 5000
            self.seed = 2512
        elif self.mode == 'validation':
            self.num_preload = 500
            self.seed = 170817
        elif self.mode == 'test':
            self.num_preload = 10000
            self.seed = torch.randint(0, 2**32, (1,)).item()
            self.seed = 4230156823#4230156823#2473530650
        
        print(f'Using seed {self.seed}')
        #torch.manual_seed(self.seed)


        # load priors
        if prior_filepath is None:
            log.info("No prior was given: loading default prior...")

        #load WaveformGenerators
        self.waveform_generator = {kind: WaveformGenerator(kind, 
                                                            signal_duration=self.signal_duration) 
                                                            for kind in self.source_kind}

        # load ASDs
        self.ASDs = {}
        for det in self.ET.arms:
            self.ASDs[det] = ASD_Sampler(det, asd_file, fs=self.fs, duration=self.signal_duration, device=self.device).asd_reference.unsqueeze(0)

        self.ASDs = TensorDict.from_dict(self.ASDs)

        # set up self random number generator
        self.rng  = torch.Generator(device)
        if not random_seed:
            random_seed = torch.randint(0, 2**32, (1,)).item()
        self.rng.manual_seed(random_seed)

        # initialize whitening
        self.WhitenNet = WhitenNet(duration=signal_duration, 
                                    fs     = self.fs,
                                    device = device,
                                    rng    = self.rng)
        
        # load prior
        self._load_prior(prior_filepath)


    @property
    def fs(self):
        return int(self.waveform_generator[self.kinds[0]].fs)
    
    @property
    def delta_t(self):
        return torch.tensor(1/self.fs)
    
    @property
    def det_names(self):
        return self.ET.arms

    @property
    def noise_std(self):
        return 1 / torch.sqrt(2*self.delta_t)
    
    @property
    def kinds(self):
        return self.source_kind

    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            self._frequencies = rfftfreq(self.fs*self.signal_duration, d=1/self.fs, device=self.device)
        return self._frequencies
    
    @property
    def freqs(self):
        if not hasattr(self, '_freqs'):
            self._freqs = rfftfreq(2*self.signal_duration*self.fs, d=self.delta_t, device=self.device)
        return self._freqs
    
    @property
    def means(self):
        if not hasattr(self, '_means'):
            self._means = dict()
            for parameter in self.inference_parameters:
                par_name, kind, _ = parameter.split('_')
                #NOTE - MORE SATANA LINES HERE!!! 
                if parameter.startswith('tcoal'):
                    par_name = 'time_shift'
                try:
                    self._means[parameter] = float(self.multivariate_prior_extrinsic.priors[par_name].mean)
                except:
                    self._means[parameter] = float(self.multivariate_prior_intrinsic[kind].priors[par_name].mean)
        return self._means
    
    @property
    def stds(self):
        if not hasattr(self, '_stds'):
            self._stds = dict()
            for parameter in self.inference_parameters:
                par_name, kind, _ = parameter.split('_')
                #NOTE - MORE SATANA LINES HERE!!! 
                if parameter.startswith('tcoal'):
                    par_name = 'time_shift'
                try:
                    self._stds[parameter] = float(self.multivariate_prior_extrinsic.priors[par_name].std)
                except:
                    self._stds[parameter] = float(self.multivariate_prior_intrinsic[kind].priors[par_name].std)
        return self._stds


    def _load_prior(self, prior_filepath=None):
        """
        Load the prior distributions specified in the json prior_filepath:
        if no filepath is given, the default one stored in the config dir will be used
        
        This function first reads the json file, then store the prior as a hyperion's MultivariatePrior instance. 
        Prior's metadata are stored as well. Metadata also contains the list of the inference parameters 
        The reference_time of the GWDetector instances is finally updated to the value set in the prior

        """
        #load the json file
        if prior_filepath is None:
            prior_filepath = CONF_DIR + '/population_priors.yml' 

        with open(prior_filepath, 'r') as yaml_file:
            prior_kwargs = yaml.safe_load(yaml_file)
            intrinsic_prior_conf = prior_kwargs['intrinsic_parameters']
            extrinsic_prior_conf = prior_kwargs['extrinsic_parameters']
            
        self._prior_metadata = prior_kwargs
        self.intrinsic_prior = intrinsic_prior_conf
        self.extrinsic_prior = extrinsic_prior_conf
                        
        self.multivariate_prior_extrinsic = MultivariatePrior(self.extrinsic_prior, device = self.device, seed = self.seed)

        # convert prior dictionary to MultivariatePrior
        self.multivariate_prior_intrinsic = {}
        for kind in self.kinds:
            self.multivariate_prior_intrinsic[kind] = MultivariatePrior(self.intrinsic_prior[kind], device = self.device, seed = self.seed)

            #add M and q to prior dictionary
            #NB: they are not added to MultivariatePrior to avoid conflict with the waveform_generator 
            #    this is intended when the inference parameters contain parameters that are combination of the default's one
            #    (Eg. the total mass M =m1+m2 or q=m2/m1 that have no simple joint distribution) 
            #    In this way we store however the metadata (eg. min and max values) without compromising the simulation 
            inference_parameters_flag = np.unique([p.split('_')[0] for p in self.inference_parameters])
            # delete double equal entries
            self.derived_parameters = [d_p for d_p in ['M', 'q', 'Mchirp'] if d_p in inference_parameters_flag]

            for p in self.derived_parameters:
                self.multivariate_prior_intrinsic[kind].add_prior({p: prior_dict_[p](self.multivariate_prior_intrinsic[kind].priors['mass1'], self.multivariate_prior_intrinsic[kind].priors['mass2'])})
                min, max = float(self.multivariate_prior_intrinsic[kind].priors[p].minimum), float(self.multivariate_prior_intrinsic[kind].priors[p].maximum)
                # add derived parameters to multivate prior metadata
                metadata = {'distribution':f'{p}_uniform_in_components', 'kwargs':{'minimum': min, 'maximum': max}}
                #self.multivariate_prior_intrinsic[kind][p] = metadata   
        
        metadata = {}
        for par in self.inference_parameters:
            par_name, kind, _ = par.split('_')
            if par_name == 'tcoal':
                metadata[par] = {'kwargs':{'minimum': str(-self.signal_duration),
                                           'maximum': '0'}}
            elif par_name == "q":
                metadata[par] = {'kwargs':{'minimum': 0.0, 
                                           'maximum':1.0}}
            else:
                try:
                    metadata[par] = {'kwargs':{'minimum':self.multivariate_prior_intrinsic[kind].priors[par_name].minimum, 
                                               'maximum':self.multivariate_prior_intrinsic[kind].priors[par_name].maximum}}
                except:
                    metadata[par] = {'kwargs':{'minimum':self.multivariate_prior_extrinsic[kind].priors[par_name].maximum, 
                                               'maximum':self.multivariate_prior_extrinsic[kind].priors[par_name].maximum}}

        
        #update reference gps time in detectors
        self.ET.set_reference_time(prior_kwargs['reference_gps_time'])

        #add inference parameters to metadata
        self._prior_metadata['inference_parameters'] = self.inference_parameters

        self._prior_metadata["parameters"] = metadata
        
        #add means and stds to metadata
        self._prior_metadata['means'] = self.means
        self._prior_metadata['stds']  = self.stds

        self._prior_metadata['fs'] = self.fs
        
        return
    
    def _compute_der_par(self, prior_samples):
        #sorting m1 and m2 so that m2 <= m1
        m1, m2 = prior_samples['mass1'], prior_samples['mass2']
        m1, m2 = q_prior._sort_masses(m1, m2)
        
        m_ch_pattern = re.compile(r'^(Mchirp_)')
        q_pattern = re.compile(r'^(q_)')
        m_pattern = re.compile(r'^(M_)')

        if any(m_ch_pattern.match(p) for p in self.inference_parameters):
            prior_samples['Mchirp'] = ((m1*m2)**(3/5)/(m1+m2)**(1/5))

        if any(q_pattern.match(p) for p in self.inference_parameters):
            prior_samples['q'] = (m2/m1)

        if any(m_pattern.match(p) for p in self.inference_parameters):
            prior_samples['M'] = (m1+m2)
        
        return prior_samples
    
    def _choose_numbers_of_signals(self, kinds = ['BNS', 'NSBH', 'BBH']):
        """Chooses the number of signals to overlap for each of the classes"""
        n_signals = {}
        for kind in kinds:
            n_signals[kind] = np.random.randint(self.max_signals[kind]+1)
        return n_signals
    
    def apply_time_shift(self, hp, hc, shifts):
        """
        Apply a time shift to the waveform.

        """
        
        hp = torch.nn.functional.pad(hp, ((self.signal_duration*self.fs), 0), 'constant', 0)
        hc = torch.nn.functional.pad(hc, ((self.signal_duration*self.fs), 0), 'constant', 0)
                
        hp_f = rfft(hp, n=hp.shape[-1], norm=self.fs)
        hc_f = rfft(hc, n=hp.shape[-1], norm=self.fs)
        
        hp_f *= torch.exp(-2j*torch.pi*self.freqs*shifts)
        hc_f *= torch.exp(-2j*torch.pi*self.freqs*shifts)
        
        hp = irfft(hp_f, n=hp.shape[-1], norm=self.fs)
        hc = irfft(hc_f, n=hp.shape[-1], norm=self.fs)
        
        #crop
        hp = hp[:, -self.signal_duration*self.fs:]
        hc = hc[:, -self.signal_duration*self.fs:]
                
        '''
        # pad, assuming is negative TODO
        # TODO - this is not the most efficient way to do it
        for i, shift in enumerate(shifts):
            dummy_p = hp[i]
            dummy_c = hc[i]

            # pad
            dummy_p = torch.nn.functional.pad(dummy_p, (0, abs(int(shift*self.fs))), 'constant', 0)
            dummy_c = torch.nn.functional.pad(dummy_c, (0, abs(int(shift*self.fs))), 'constant', 0)

            # crop
            dummy_p = dummy_p[-hp.shape[-1]:]
            dummy_c = dummy_c[-hc.shape[-1]:]

            hp[i] = dummy_p
            hc[i] = dummy_c
        '''


        return hp, hc


    def get_kind_batch(self, kind, n_signals):

        #sample extrinsic parameters
        e_pars = self.multivariate_prior_extrinsic.sample((n_signals, self.batch_size, 1))
        snr = e_pars['snr']
        snr_sorted, _ = snr.sort(dim=0, descending=True)
        
        out_prior_samples = {}

        #get the corresponding preloaded waveforms
        #projected = TensorDict.from_dict({det: torch.zeros(self.batch_size, self.signal_duration*self.fs, device=self.device) for det in self.ET.arms})

        projected = []

        if self.mode == 'test':
            tcoals = []
            durations = []

        for i in range(n_signals):
            idxs = self.get_idxs()
            hp = self.preloaded_wvfs[kind][idxs]['hp']
            hc = self.preloaded_wvfs[kind][idxs]['hc']

            ra  = e_pars['ra'][i]
            dec = e_pars['dec'][i]
            polarization = e_pars['polarization'][i]
            snr = snr_sorted[i]
            
            # time shift
            time_shift = e_pars['time_shift'][i]
            # if self.mode == 'test':
            #     if i != 0 :
            #         #print('lo sto facendo')
            #         time_shift = torch.normal(e_pars['time_shift'][0], 0.5)
            #         time_shift = time_shift.clamp(-10, 0)
            #         #print(f'time_shift is {time_shift}')

            # TODO - aasign time shift to  signal before summing
            hp, hc = self.apply_time_shift(hp, hc, time_shift)
            
            #project the signal onto the detector
            projected_i = self.ET.project_wave(hp, hc, ra, dec, polarization) 

            # REVIEW - control this, also if can work in batches
            projected_i = rescale_to_network_snr(projected_i, snr, self.ASDs, self.signal_duration, self.fs)
            projected.append(projected_i)

            # adjust output
            for p in self.inference_parameters:
                if p.endswith(f'{kind}_{i+1}'):
                    par_name = p.replace(f'_{kind}_{i+1}', '')
                    if par_name in e_pars.keys():
                        out_prior_samples[p] = e_pars[par_name][i].squeeze(-1)

                    elif par_name=='tcoal':
                        #print(f'tcoal_{kind}_{i+1} is {self.prior_samples[kind]["tcoal"][idxs]}')
                        #print(f'time_shift is {time_shift.squeeze()}')
                        out_prior_samples[p] = self.prior_samples[kind]['tcoal'][idxs] + time_shift.squeeze()  # REVIEW
                        #print(f'out tcoal_{kind}_{i+1} is {out_prior_samples[p]}')
                    # select the right parameter by removing the kind and the signal number
                    else:
                        out_prior_samples[p] = self.prior_samples[kind][idxs][par_name]

            if self.mode == 'test':
                # t coals of every kind
                tcoals_i = {key: out_prior_samples[key] for key in out_prior_samples.keys() if key.startswith('tcoal')}
                tcoals.append(tcoals_i)  # FIXME non si updata e rimette tutti tutte le volte

                # duration of each template
                if 'Mchirp' in self.prior_samples[kind].keys():
                    mchirp = self.prior_samples[kind]['Mchirp'][idxs]
                else:
                    M = self.prior_samples[kind]['M'][idxs]
                    q = self.prior_samples[kind]['q'][idxs]
                    m1 = M / (1 + q)
                    m2 = M - m1
                    mchirp = ((m1*m2)**(3/5)/(m1+m2)**(1/5))

                f_low = self._prior_metadata['f_lower'][kind]

                durations_i = 2.2 * ((1.21 / mchirp) ** (5 / 3)) * ((100 / f_low) ** (8 / 3))
                durations.append(durations_i)

        if self.mode == 'test':    
            return projected, out_prior_samples, snr_sorted, tcoals, durations

        return projected, out_prior_samples
    
    def standardize_parameters(self, prior_samples):
        """Standardize prior samples to zero mean and unit variance"""
        
        out_prior_samples = []
        for parameter in self.inference_parameters:
            par_name, kind, _ = parameter.split('_')
            # le righe successiva sono satana
            if par_name == 'tcoal':
                par_name = 'time_shift'
                
            if par_name in self.multivariate_prior_extrinsic.priors.keys():
                standardized = self.multivariate_prior_extrinsic.priors[par_name].standardize_samples(prior_samples[parameter])
            else:  
                standardized = self.multivariate_prior_intrinsic[kind].priors[par_name].standardize_samples(prior_samples[parameter])
            out_prior_samples.append(standardized.unsqueeze(-1))
            
        out_prior_samples = torch.cat(out_prior_samples, dim=-1)
        return out_prior_samples
    
    def get_idxs(self):
        if not hasattr(self, 'preloaded_wvfs'):
            raise ValueError('There are no preloaded waveforms. Please run pre_load_waveforms() first.')

        idxs = torch.arange(self.num_preload).float()
        return torch.multinomial(idxs, self.batch_size, replacement=False)
    
# ______________________________________________________________________________________________________________________
    def preload_waveforms(self):
        """
        Preload a set of waveforms to speed up the generation of the dataset.
        """
        
        
        self.preloaded_wvfs = {}
        self.tcoal = {}
        self.prior_samples = {}
        for kind in self.kinds:
            log.info(f'Preloading a new set of {kind} waveforms...')
            #first we sample the intrinsic parameters
            self.prior_samples[kind] = self.multivariate_prior_intrinsic[kind].sample(self.num_preload)
            
            der_inf_parameter_pattern = re.compile(r'^(M_|q_|Mchirp_)')

            if any(der_inf_parameter_pattern.match(p) for p in self.inference_parameters):
                self.prior_samples[kind] = self._compute_der_par(self.prior_samples[kind])
                        
            #then we call the waveform generator
            hp, hc, tcoal = self.waveform_generator[kind](self.prior_samples[kind].to('cpu'), npool=self.npool)
            
            log.info(f'Done generating {kind} waveforms...')

            #store the waveforms as a TensorDict
            wvfs = {'hp': hp, 'hc': hc}

            self.preloaded_wvfs[kind] = TensorDict.from_dict(wvfs).to(self.device)
            
            # put in prior samples
            self.prior_samples[kind]['tcoal'] = tcoal.squeeze()

        return
# ______________________________________________________________________________________________________________________
    
    def __getitem__(self, idx=None, add_noise=True, priors_samples=None):
        # sample the number of signals to overlap for each class
        if self.fixed_signals:
            # TODO: fixed signal to be a dictionary to handle different number of signals for each kind
            n_signals = {kind: self.num_fixed_signals for kind in self.kinds}

        else:    
            n_signals = self._choose_numbers_of_signals()

        out_prior_samples = {}
        # DIM OF THE THING IS (BATCH_SIZE, SIGNAL_DURATION*FS)
        projected_summed = TensorDict.from_dict({det: torch.zeros(self.batch_size, self.signal_duration*self.fs, device=self.device) 
                                                 for det in self.ET.arms})
        for kind in self.kinds:
            
            if self.mode == 'test':
                projected_i, out_prior_kind, snr, tcoals, durations = self.get_kind_batch(kind, n_signals[kind])
                for det in self.ET.arms:
                    for i in range(len(projected_i)):
                        projected_summed[det] += projected_i[i][det]

            else:
                projected_i, out_prior_kind = self.get_kind_batch(kind, n_signals[kind])
                for det in self.ET.arms:
                    for i in range(len(projected_i)):
                        projected_summed[det] += projected_i[i][det]  # summing the signals on the different channels for each kind
                
            out_prior_samples.update(out_prior_kind)

        # standardize the prior samples
        out_prior_samples = self.standardize_parameters(out_prior_samples)

        # adding noise + whitening
        #NOTE - we Whiten now with the "gwpy" method in HYPERION which is more accurate than the "pycbc" one
        whitened = self.WhitenNet(h         = projected_summed, 
                                  asd       = self.ASDs,
                                  add_noise = self.add_noise,
                                  method    = 'gwpy',
                                  normalize = True)
        #convert to a single float tensor
        out_whitened_strain = torch.stack([whitened[det] for det in self.ET.arms], dim=1).float()


        if self.plot_template:

            # Ensure the data is transferred once and stays on CPU for all subsequent operations
            projected_summed_cpu = {det: projected_summed[det].double().cpu() for det in self.ET.arms}
            #torch.cuda.synchronize()

            times = np.arange(-self.signal_duration, 0, 1/self.fs)

            fig, axs = plt.subplots(3, 1, figsize=(20, 20))
            for i, det in enumerate(self.ET.arms):
                #plt.figure()
                proj_numpy = projected_summed_cpu[det].numpy()[0] 
                axs[i].plot(times, proj_numpy)
                axs[i].set_title(det, fontsize=24)
                axs[i].set_xlabel('Time [s]', fontsize=20)  
                axs[i].set_ylabel('Strain', fontsize=20)
                axs[i].tick_params(axis='both', labelsize=18)
                for j in range(n_signals['BBH']):
                    axs[i].axvline(out_prior_kind[f'tcoal_BBH_{j+1}'].cpu().numpy()[0], color='red')

            plt.tight_layout()
            plt.savefig('training_results/template.pdf')

        if self.plot_signal_noise:

            fig, axs = plt.subplots(3, 1, figsize=(12, 12))
            for i, det in enumerate(self.ET.arms):
                axs[i].plot(out_whitened_strain[0, i].cpu().numpy())
                axs[i].set_title(det, fontsize=24)
                axs[i].set_xlabel('Time [s]', fontsize=20)  
                axs[i].set_ylabel('Strain', fontsize=20)
                axs[i].tick_params(axis='both', labelsize=18)
            plt.savefig('training_results/signal_noise.pdf')

        if self.mode == 'test':
            return out_prior_samples, out_whitened_strain, projected_i, self.ASDs, snr, tcoals, durations

        if self.mode != 'test':
            return out_prior_samples, out_whitened_strain, self.ASDs
   
"""A waveform generator class"""
import os
import h5py
import yaml
import torch
import numpy as np
import pandas as pd
import multiprocess as mp

from tqdm import tqdm
from pycbc.waveform import get_td_waveform
from hyperion.core.distributions import MultivariatePrior

from ..config import CONF_DIR

#mp.set_start_method('spawn', force=True) # It only works with 'spawn' method when doing inference


class WaveformGenerator():
    
    def __init__(self, source_kind, signal_duration, prior_filepath=None):
        
        assert source_kind in ['BBH', 'BNS', 'NSBH'], 'source_kind must be one of "BBH", "BNS", or "NSBH"'
        
        self.kind = source_kind
        self.duration = signal_duration
        self._load_prior(prior_filepath)

        self.times_array = np.linspace(int(-self.duration), 0, int(self.duration*self.fs))
        
        self.delta_t = 1/self.fs

        return
    
    @property
    def fs(self):
        if not hasattr(self, '_fs'):
            self._fs = 1/self.pycbc_args['delta_t']
        return self._fs
    
    
    
    def _load_prior(self, prior_filepath):
        
        if prior_filepath is None:
            prior_filepath = os.path.join(CONF_DIR, 'population_priors.yml')
        
        with open(prior_filepath, 'r') as f:
            prior_config = yaml.safe_load(f)
            
        self.pycbc_args = {'delta_t': 1/prior_config['fs'], 
                           'f_lower': prior_config['f_lower'][self.kind],
                           'approximant': prior_config['approximants'][self.kind]
                           }
                
    
    def _adjust_signal_duration(self, hp, hc):
        """Crops (or pads) the signal to the desired duration"""
        
        template_duration = hp.shape[-1]/self.fs
        times_np = hp.sample_times.numpy()
        hp_np = hp.numpy()
        hc_np = hc.numpy()

        #signal is longer than desired length
        #we chose randomly a window of the desired length
        if template_duration > self.duration:
            hp_np = hp_np[int(-self.duration*self.fs):]
            hc_np = hc_np[int(-self.duration*self.fs):]
            t_wvf = times_np[int(-self.duration*self.fs):]

        #signal is shorter than desired length
        #we pad the signal with zeros
        elif template_duration <= self.duration:
            pad = int((self.duration - template_duration)*self.fs)
            hp_np = np.pad(hp_np, (pad, 0))
            hc_np = np.pad(hc_np, (pad, 0))
            times_np_left = np.linspace(-self.delta_t*pad, 0, pad) + times_np.min()
            t_wvf = np.concatenate([times_np_left, times_np])
            
        # this is for matching the zeros of the waveform 0 to the window 0, very briefly
        time_to_merger = np.interp(0, t_wvf, self.times_array)
    
        return torch.from_numpy(hp_np), torch.from_numpy(hc_np), time_to_merger
    
    
    def generate_waveform(self, i=None, parameters=None):
        
        parameters = self.prior_samples[i]
        
        #manage lambda parameters
        lambda_pars = {'lambda1': None, 'lambda2': None}
        for p in lambda_pars.keys():
            if p in self.prior_samples.keys():
                lambda_pars[p] = float(parameters.pop(p))
        
    
        hp, hc = get_td_waveform(**self.pycbc_args, **parameters, **lambda_pars)
        hp, hc, time_to_merger = self._adjust_signal_duration(hp, hc)
        
        return hp, hc, time_to_merger
    
    def save_waveform(self, hp, hc, i, output_directory):
        
        filepath = os.path.join(output_directory, f'{self.kind}_waveform_{i}.h5')
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('hp', data=hp.numpy())
            f.create_dataset('hc', data=hc.numpy())
            f.create_dataset('duration', data=hc.duration)
            f.create_dataset('fs', data=self.fs)
            f.create_dataset('epoch', data=float(hp.start_time))
            
    def save_prior_samples(self, output_directory):
        filepath = os.path.join(output_directory, f'{self.kind}_prior_samples.csv')
        prior_df = pd.DataFrame.from_dict(self.prior_samples.to_dict())
        prior_df.to_csv(filepath, index=False)
        return
        
        
    def __call__(self, prior_samples, N=10, save=False, output_directory=None, npool=None):
        
        if save:
            assert output_directory is not None, 'output_directory must be provided if save is True'
            out_dir = os.path.join(output_directory, self.kind)
            os.makedirs(out_dir, exist_ok=True)
        
        self.prior_samples = prior_samples
        N = len(prior_samples)
        
        if save:
            self.save_prior_samples(output_directory)
            print('\n[INFO]: prior samples saved\n')
        
        iteration = 0
        hps = []  # plurals
        hcs = []  # plurals
        tcoals = []  # plurals

        npool = os.cpu_count() if npool is None else npool
        with mp.Pool(npool) as p:
            for hp, hc, time_to_merger in tqdm(p.imap(self.generate_waveform, range(N)), total = N, ncols = 100, ascii=' ='):
                
                # creating lists in preparation for tensors
                hps.append(hp)
                hcs.append(hc)
                tcoals.append(time_to_merger)
                
                if save:
                    self.save_waveform(hp, hc, iteration, out_dir)
                iteration+=1

            # creates the tensor with all signals
            hps = torch.stack(hps)
            hcs = torch.stack(hcs)
            tcoals = torch.tensor(tcoals).unsqueeze(-1).float()
        
        return hps, hcs, tcoals
        
        
    
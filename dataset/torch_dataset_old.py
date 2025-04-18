"""Class for data loading and preprocessing"""

import os
import yaml
import h5py
import torch
import numpy as np

from tqdm import tqdm
from pycbc.types import TimeSeries
from torch.utils.data import Dataset

from hyperion.training import ASD_Sampler
from hyperion.core.fft import rfft, irfft
from hyperion.core.distributions import MultivariatePrior, prior_dict_

from ..config import CONF_DIR
from ..simulations import (EinsteinTelescope, 
                           WaveformGenerator,
                           rescale_to_network_snr, 
                           luminosity_distance_from_redshift)

import matplotlib.pyplot as plt

asd_file = os.path.join(CONF_DIR, 'ASD_curves', 'ET_MDC_asd.txt')

class DatasetGenerator(Dataset):
    
    def __init__(self, 
                 data_dir, 
                 duration=128, 
                 noise_duration=2*128, 
                 max_signals = {'BNS' : 1, 
                                'NSBH': 1,
                                'BBH' : 2},
                 source_kind = ['BNS', 'NSBH', 'BBH'],
                 train_split = 0.9, 
                 mode = 'training',
                 fixed_signals = False,
                 num_fixed_signals = 2,
                 **ET_kwargs
                                ):


        self.data_dir = data_dir
        self.duration = duration
        self.noise_duration = noise_duration
        self.source_kind = source_kind
        self.fixed_signals = fixed_signals
        self.num_fixed_signals = num_fixed_signals

        #get data files according to trainig or validation mode and splitting
        self.data_files = {}
        for kind in self.source_kind:
            files = os.listdir(os.path.join(data_dir, kind))
            if mode == 'training' or mode == 'testing':
                self.data_files[kind] = files[:int(train_split*len(files))]
                
            elif mode == 'validation':
                self.data_files[kind] = files[int(train_split*len(files)):]
        
        #load extrinsic prior
        with open(os.path.join(CONF_DIR, 'population_priors.yml'), 'r') as f:
            prior_conf = yaml.safe_load(f)
            intrinsic_prior_conf = prior_conf['intrinsic_parameters']
            extrinsic_prior_conf = prior_conf['extrinsic_parameters']
            self.fs = prior_conf['fs']
            
        #extrinsic parameter prior    
        self.extrinsic_prior = MultivariatePrior(extrinsic_prior_conf)
        
        #redshift prior
        self.redshift_prior = {}
        for kind in self.source_kind:
            dist = intrinsic_prior_conf[kind]['redshift']['distribution']
            kwargs = intrinsic_prior_conf[kind]['redshift']['kwargs']
            self.redshift_prior[kind] = prior_dict_[dist](**kwargs)
        

        #max number of signals to overlap
        self.max_signals = max_signals
        
        #load Einstein Telescope detector
        self.ET = EinsteinTelescope(**ET_kwargs)

        
        #load ASDs
        self.ASDs = {}
        for det in self.ET.arms:
            self.ASDs[det] = ASD_Sampler(det, asd_file, fs=self.fs, duration=self.duration).asd_reference


        #load WaveformGenerators
        base_seed = 123 if mode == 'training' else 1234
        self.waveform_generators = {kind: WaveformGenerator(kind, seed=base_seed) for kind in self.source_kind}

        super().__init__()
        return
    
    def __len__(self):
        return 1*sum([len(self.data_files[kind]) for kind in self.data_files])
    
    @property
    def kinds(self):
        return self.source_kind
    
    @property
    def PSDs(self):
        if not hasattr(self, '_PSDs'):
            self._PSDs = {det: self.ASDs[det]**2 for det in self.ET.arms}
        return self._PSDs
    
    @property
    def delta_t(self):
        return torch.as_tensor(1/self.fs)
    
    @property
    def noise_mean(self):
        if not hasattr(self, '_noise_mean'):
            self._noise_mean = torch.zeros(self.duration*self.fs)
        return self._noise_mean
    
    
    @property
    def noise_std(self):
        if not hasattr(self, '_noise_std'):
            self._noise_std = 1 / torch.sqrt(2*self.delta_t)
        return self._noise_std
    
    #=================
    # output options
    #=================
    @property
    def plot(self):
        return self._plot
    @plot.setter
    def plot(self, value):
        self._plot = value
        
    @property
    def add_noise(self):
        return self._add_noise
    @add_noise.setter
    def add_noise(self, value):
        self._add_noise = value
        
    @property
    def means(self):
        return {kind: 0 for kind in self.kinds}
    
    @property
    def stds(self):
        return {kind: 1 for kind in self.kinds}


    @property
    def prior_metadata(self):

        prior_metadata = {'inference_parameters': ['BNS', 'NSBH', 'BBH'],
                                    'means': self.means,
                                    'stds': self.stds,
                          'parameters':{}
                    }
        for name in self.kinds:
            prior_metadata['parameters'][name] = {'kwargs':{'minimum': 0, 'maximum': self.max_signals[name]}}
        return prior_metadata


    def read_data(self, idx, kind):
        """Reads the plus and cross polarizations of the waveform and the time array from the data file
    
        Args:
        -----
            idx (int): index of the file
            kind (str): kind of signal (BNS, NSBH, BBH)
            
        Returns:    
        --------
            hp (pycbc TimeSeries): plus polarization
            hc (pycbc TimeSeries): cross polarization
        """

        fpath = os.path.join(self.data_dir, kind, self.data_files[kind][idx])
        with h5py.File(fpath, 'r') as hf:
            #hf = self.data_files[kind][idx]
            #duration = hf['duration'][()]
            fs = hf['fs'][()]            
            hp = np.array(hf['hp'])
            hc = np.array(hf['hc'])
            epoch = hf['epoch'][()]
        
        hp = TimeSeries(hp, delta_t=1/fs, epoch = epoch)
        hc = TimeSeries(hc, delta_t=1/fs, epoch = epoch)
 
        return hp, hc
    
    
    def _choose_numbers_of_signals(self, kinds = ['BNS', 'NSBH', 'BBH']):
        """Chooses the number of signals to overlap for each of the classes"""
        n_signals = {}
        for kind in kinds:
            n_signals[kind] = np.random.randint(self.max_signals[kind]+1)
        return n_signals
    
    
    def _adjust_signal_duration(self, hp, hc):
        """Crops (or pads) the signal to the desired duration"""
        
        duration = hp.shape[-1]/self.fs
        time_shift = 0 #gets updated only in the elif 
        
        #signal is longer than desired length
        #we chose randomly a window of the desired length
        if duration > self.duration:
            win_len = 2*self.duration*self.fs
            #iw = np.random.randint((hp.shape[-1]-win_len)//2, hp.shape[-1]-win_len)
            #min_t = min(hp.start_time, hp.duration-2*self.duration)
            #iw = np.random.uniform(min_t, hp.end_time-self.duration)
            #hp = hp.resize(2*self.duration*self.fs)
            #hc.resize(2*self.duration*self.fs)
            
            #iw = hp.shape[-1]-win_len
            #hp = hp[iw : iw+win_len]
            #hc = hc[iw : iw+win_len]
            #hp = hp.time_slice(hp.end_time-self.duration, hp.end_time, mode = 'nearest')
            #hc = hc.time_slice(hc.end_time-self.duration, hc.end_time, mode = 'nearest')
            hp.append_zeros(self.duration*self.fs)
            hc.append_zeros(self.duration*self.fs)

            

        #signal is shorter than desired length
        #we pad the signal with zeros
        elif duration <= self.duration:
           
            
            pad = 2*self.duration * self.fs
            hp.append_zeros(pad)
            hc.append_zeros(pad)
            hp.prepend_zeros(pad)
            hc.prepend_zeros(pad)

            '''

            hp.resize(self.duration*self.fs)
            hc.resize(self.duration*self.fs)
            
            max_time_shift = hp.sample_times[-1]#self.duration - hp.duration
            time_shift = np.random.uniform(0, max_time_shift)
            
            hp = hp.cyclic_time_shift(time_shift)
            hc = hc.cyclic_time_shift(time_shift)
            #print('new time to merger', -hp.sample_times.numpy()[-1])

            #print('new time to merger', -hp.sample_times.numpy()[-1])
            '''
        central_time = 0.9*np.random.uniform(-self.duration/2, self.duration/2)

        hp = hp.time_slice(central_time - self.duration/2, central_time + self.duration/2, mode = 'nearest')
        hc = hc.time_slice(central_time - self.duration/2, central_time + self.duration/2, mode = 'nearest')
        # the minus sign account for the fact that pycbc places the merger at t=0
        # hence with this convention t<0 means that the merger has occurred
        # while t>0 means that the merger will occur in the future
        time_to_merger = -hp.sample_times.numpy()[-1] #+ time_shift
            
        return torch.from_numpy(hp.numpy()), torch.from_numpy(hc.numpy()), time_to_merger
   
   
    def get_waveform(self, idx, kind, gps_time):
        hp, hc = self.read_data(idx, kind)
        #hp, hc = self.waveform_generators[kind].generate_waveform()
        
        #sample redshift from prior
        redshift = self.redshift_prior[kind].sample(1).item()
    
        #rescale the signal with luminosity distance from redshift
        if kind =='BBH':
            dL = luminosity_distance_from_redshift(redshift)
        else:
            dL = np.random.uniform(100, 3000)
        #dL = luminosity_distance_from_redshift(redshift)
        hp/=dL
        hc/=dL
        
        #rescale the waveform to the desired duration
        hp, hc, time_to_merger = self._adjust_signal_duration(hp, hc)
        
        #sample extrinsic parameters
        e_pars = self.extrinsic_prior.sample(1)
        ra = e_pars['ra'].item()
        dec = e_pars['dec'].item()
        polarization = e_pars['polarization'].item()
        
        #project the signal onto the detector
        projected = self.ET.project_wave(hp, hc, ra, dec, polarization, gps_time)        
        return projected, time_to_merger
    
    
    def _add_gaussian_noise(self, h):
        """
        Adds gaussian noise to the whitened signal(s).
        To ensure that noise follows a N(0, 1) distribution we divide by the noise standard deviation
        given by 1/sqrt(2*delta_t) where delta_t is the sampling interval.
        
        Args:
        -----
            h (dict of torch.Tensor): whitened signals
        """
        
        
        
        for det in self.ET.arms:
            noise = torch.normal(mean=self.noise_mean, std=self.noise_std)
            h[det] += noise
            h[det] /= self.noise_std
        
        return h
    
    def standardize(self, num, kind):
        return (num - self.means[kind])/self.stds[kind]

    
    def __getitem__(self, idx=None):

        #sample the reference GPS time
        gps_time = self.extrinsic_prior.priors['gps_time'].sample(1).item()
        

        #sample the number of signals to overlap for each class
        if self.fixed_signals:
            n_signals = {kind: self.num_fixed_signals for kind in self.kinds}

        else:    
            n_signals = self._choose_numbers_of_signals()
        
        #sum of all signals
        #h_sum = {det: torch.zeros(self.duration*self.fs) for det in self.ET.arms}

        #individual signals
        # it is a nested dictionary with the following structure
        # KIND -> DETECTOR -> SIGNAL
        h_i = {kind: {det: torch.zeros((self.max_signals[kind], self.duration*self.fs)) 
                      for det in self.ET.arms} 
                      for kind in self.kinds}
        
        time_to_mergers = {kind: [-2*self.duration for _ in range(self.max_signals[kind])] 
                                  for kind in self.kinds}
        
        for kind in ['BNS', 'NSBH', 'BBH']:
            for i in range(n_signals[kind]):
                
                idx = np.random.randint(len(self.data_files[kind]))

                hi, t = self.get_waveform(idx, kind, gps_time)
                
                for det in hi:
                    h_i[kind][det][i] = hi[det]
                    #h_sum[det] += hi[det]
                
                time_to_mergers[kind][i] = t
                
        '''
        if self.add_noise:
            h_sum = self.whiten(h_sum)
            h_sum = self._add_gaussian_noise(h_sum)

        if self.plot:
            t_array = np.linspace(0, self.duration, self.duration*self.fs)
            plt.figure()
            for det in h_sum:
                plt.plot(t_array, h_sum[det].cpu().numpy())
            plt.title(f'all signal')
            plt.xlabel('time [s]')
            plt.ylabel('strain')
            plt.savefig('prova.png', dpi=200)
            plt.show()       
        '''
        
        n_signals_class = dict()
        for kind in self.kinds:
            n_signals_class[kind] = torch.zeros(self.max_signals[kind]+1)
            n_signals_class[kind][n_signals[kind]-1] = 1

        #out_n = torch.tensor([self.standardize(n_signals[kind], kind) for kind in self.kinds]).float()
        out_n = torch.tensor([n_signals[kind] for kind in self.kinds]).float()
        #print(out_n)
    
        return h_i, out_n, time_to_mergers

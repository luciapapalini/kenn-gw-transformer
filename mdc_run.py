import os
import gwpy
import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from deepfastet.model import Kenn, KennConfig, KennAttention, KennAttentionHead
from deepfastet.training import Trainer
from torch.utils.data import DataLoader
from deepfastet.dataset import DatasetGenerator, WhitenNet
from hyperion.training.train_utils import get_optimizer, get_LR_scheduler

from hyperion.core.flow import Flow, CouplingTransform, AffineCouplingLayer, RandomPermutation
from hyperion.core.distributions import MultivariateNormalBase

from hyperion.core import PosteriorSampler


def analyze_signal(t_gps, kind):
    
    ifile = int((t_gps-1e9)/2048)
    
    strain = [TimeSeries.read(f"{MDC_directory}/{det}/{MDC_files[det][ifile]}", 
                               format = 'gwf',
                               channel = f"{det}:STRAIN")
               .crop(t_gps-DURATION, t_gps+DURATION)
               .whiten()
               .resample(FS)
               .crop(t_gps-DURATION/2, t_gps+DURATION/2)
               .value
            for det in et_arms
            ]
    strain = torch.stack([torch.from_numpy(s.copy()*noise_std) for s in strain]).unsqueeze(0).to(device).float()
    
    posterior = sampler.sample_posterior(num_samples=5000, strain=strain, 
                                         post_process=False, verbose=False)
    
    
    
    
    
    
    return










if __name__ == '__main__':
    
    SNR_threshold = 100
    
    
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = f'cuda:{num_gpus-1}'
    else:
        device = 'cpu'
    
    FS = 1024
    DURATION = 32
    
    delta_t = 1/FS
    noise_std = 1 / np.sqrt(2*delta_t)
    

    CHECKPOINT_PATH = 'training_results/BEST_trained_KENN/trained_KENN.pt'
    dataset_kwargs = {'data_dir': '../gwdata-archive/ET_mdc/test_dataset',
                        'duration': DURATION, 
                        'max_signals':{'BNS'  : 1, 
                                        'NSBH': 1,
                                        'BBH' : 2},  
                        'train_split':0.9,
                        }
        
    test_dataset   = DatasetGenerator(mode='testing', **dataset_kwargs)

    
    
    
    #read injection file
    injection_file = 'bin/tests/list_mdc1.txt'
    injection_df = pd.read_csv(injection_file, sep=' ')
    
    BBH_injections  = injection_df[injection_df['type']==3]
    NSBH_injections = injection_df[injection_df['type']==2]
    BNS_injections  = injection_df[injection_df['type']==1]

    
    #list mdc dataset files
    et_arms = ['E1', 'E2', 'E3']
    MDC_files = {det: os.listdir(f'../gwdata-archive/ET_mdc/et-origin.cism.ucl.ac.be/MDC1/data/{det}') for det in et_arms}
    MDC_directory = '../gwdata-archive/ET_mdc/et-origin.cism.ucl.ac.be/MDC1/data/'
    #initialize model
    """
    ========================================================================================
                                SETUP MODEL AND SAMPLER
    ========================================================================================
    """
    with torch.device(device):
        whitener = WhitenNet(fs=FS, 
                            duration=DURATION, 
                            device=device)
        
        #setup model 
        config = KennConfig(
                    KennAttentionHead,
                    KennAttention,
                    samples=DURATION*FS,
                    duration_in_s = DURATION,  # s
                    sampling_rate=FS,  # Hz
                    n_channels=3,
        )
        #setup model
    
        kenn = Kenn(config).float()

        base = MultivariateNormalBase(dim=config.n_channels, trainable=False)

        #COUPLING TRANSFORM ----------------------------------------------------------------
        coupling_layers = []
        for i in range(6):
            
            coupling_layers += [RandomPermutation(num_features=config.n_channels)]

            coupling_layers += [AffineCouplingLayer(num_features = config.n_channels,
                                                    num_identity = 1, 
                                                    num_transformed = 2)]
            
        coupling_transform = CouplingTransform(coupling_layers)

    
        #FLOW --------------------------------------------------------------------------------------
        state_dict = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
        flow = Flow(base_distribution = base, 
                    transformation    = coupling_transform, 
                    embedding_network = kenn,
                    prior_metadata    = test_dataset.prior_metadata, 
                    ).float()
        
        flow.load_state_dict(state_dict['model_state_dict'])

        sampler = PosteriorSampler(flow=flow, 
                                   device=device, 
                                   output_dir = CHECKPOINT_PATH[:-16])

        """
        =======================================================================================
                                            ANALYZE SIGNALS                                     
        =======================================================================================
        """
    
        #BBH
        SNR_BBH = BBH_injections['snrET_Opt'].to_numpy()
        mask    = SNR_BBH>SNR_threshold
        SNR_BBH = SNR_BBH[mask]
        
        tc = BBH_injections['tc'].to_numpy()[mask]
        
        ind = np.argsort(SNR_BBH)
        SNR_BBH = SNR_BBH[ind]
        tc = tc[ind]
        
        
        for t_gps in tqdm(tc):
            
            try:   
                analyze_signal(t_gps, kind='BBH')
            except:
                pass
        
    
    


    
    
    
    
    
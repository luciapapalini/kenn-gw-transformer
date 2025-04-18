import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set_context("talk")

from deepfastet.model import Kenn, KennConfig, KennAttention, KennAttentionHead
from deepfastet.dataset import DatasetGenerator

from hyperion.training import Trainer
from hyperion.training.train_utils import get_optimizer, get_LR_scheduler
from hyperion.core.flow import Flow, CouplingTransform, AffineCouplingLayer, RandomPermutation
from hyperion.core.distributions import MultivariateNormalBase
from hyperion.core import PosteriorSampler

from tensordict import TensorDict

from scipy.stats import norm

if __name__ == '__main__':
    
    BATCH_SIZE = 1
    DURATION   = 16
    
    INITIAL_LEARNING_RATE = 1e-4
    
    #set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.device(device):

        # inference parameters flags
        inference_parameters_flags = ['M', 'Mchirp', 'q', 'tcoal']

        # number of overlapping signals
        num_overlapping_signals = {'BBH': 2, 'NSBH': 0, 'BNS': 0}

        inference_parameters = []
        # inference parameters
        for kind in num_overlapping_signals.keys():
            if num_overlapping_signals[kind] > 0:
                for i in range(num_overlapping_signals[kind]):
                    for par in inference_parameters_flags:
                        inference_parameters.append(f'{par}_{kind}_{i+1}')
        
        #setup Dataset & Dataloader
        print('[INFO]: Using device {}'.format(device))
        
        # dataset generator arguments
        dataset_kwargs = {
            'signal_duration'     : DURATION,
            'batch_size'          : BATCH_SIZE,
            'source_kind'         : ['BBH'],
            'fixed_signals'       : True,
            'num_fixed_signals'   : num_overlapping_signals['BBH'],
            'device'              : device,
            'inference_parameters': inference_parameters,
            'plot_template'       : True,
            'plot_signal'         : True,
            'add_noise'           : True
                        }
        
        test_dataset = DatasetGenerator(mode='test', **dataset_kwargs)
        test_dataset.preload_waveforms()
                    
    
        #setup model 
        config = KennConfig(
                    KennAttentionHead,
                    KennAttention,
                    samples=DURATION*test_dataset.fs,
                    duration_in_s = DURATION,  # s
                    sampling_rate=test_dataset.fs,  # Hz
                    n_channels=3,
                    max_num_bbh=test_dataset.max_signals['BBH']+1,
                    max_num_nsbh=test_dataset.max_signals['NSBH']+1,
                    max_num_bns=test_dataset.max_signals['BNS']+1,

        )

        dim_flow_layer = len(inference_parameters) * 3 

        with torch.device(device):
            kenn = Kenn(config).float()

            base = MultivariateNormalBase(dim=len(inference_parameters), trainable=False)

            #COUPLING TRANSFORM ----------------------------------------------------------------
            coupling_layers = []
            for i in range(dim_flow_layer):
                
                coupling_layers += [RandomPermutation(num_features=len(inference_parameters))]

                coupling_layers += [AffineCouplingLayer(num_features    = len(inference_parameters),
                                                        num_identity    = len(inference_parameters)//2, 
                                                        num_transformed = len(inference_parameters)//2)]
                
            coupling_transform = CouplingTransform(coupling_layers)

            #FLOW --------------------------------------------------------------------------------------
            flow = Flow(base_distribution = base, 
                        transformation    = coupling_transform,
                        embedding_network = kenn,
                        prior_metadata    = test_dataset._prior_metadata,
                        ).float().to(device)


            #print(model)
        
        
        #load trained weights
        checkpoint_dir = os.path.join('training_results', 'trained_KENN')
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_filepath = os.path.join(checkpoint_dir, 'trained_KENN.pt')
        state_dict = torch.load(checkpoint_filepath, map_location=torch.device('cpu'))
        flow.load_state_dict(state_dict['model_state_dict'])
        
        #setup sampler
        sampler = PosteriorSampler(
            flow       = flow,
            device     = device,
            output_dir = checkpoint_filepath[:-16])
        
        parameters, strain, asd = test_dataset.__getitem__()

        print(parameters)

        zeros = torch.zeros(1, 4).to(device)
        parameters = torch.cat((parameters, zeros), dim=1).to(device)

        # inference parameters flags
        inference_parameters_flags = ['M', 'Mchirp', 'q', 'tcoal']

        print('Im before the second part')

        # number of overlapping signals
        num_overlapping_signals = {'BBH': 3, 'NSBH': 0, 'BNS': 0}

        inference_parameters = []
        # inference parameters
        for kind in num_overlapping_signals.keys():
            if num_overlapping_signals[kind] > 0:
                for i in range(num_overlapping_signals[kind]):
                    for par in inference_parameters_flags:
                        inference_parameters.append(f'{par}_{kind}_{i+1}')
        
        #setup Dataset & Dataloader
        print('[INFO]: Using device {}'.format(device))                    
    
        #setup model 
        config_2 = KennConfig(
                    KennAttentionHead,
                    KennAttention,
                    samples=DURATION*test_dataset.fs,
                    duration_in_s = DURATION,  # s
                    sampling_rate=test_dataset.fs,  # Hz
                    n_channels=3,
                    max_num_bbh=test_dataset.max_signals['BBH']+1,
                    max_num_nsbh=test_dataset.max_signals['NSBH']+1,
                    max_num_bns=test_dataset.max_signals['BNS']+1,

        )

        dim_flow_layer = len(inference_parameters) * 3 

        kenn_2 = Kenn(config_2).float()

        base_2 = MultivariateNormalBase(dim=len(inference_parameters), trainable=False)

        #COUPLING TRANSFORM ----------------------------------------------------------------
        coupling_layers_2 = []
        for i in range(dim_flow_layer):
            
            coupling_layers_2 += [RandomPermutation(num_features=len(inference_parameters))]

            coupling_layers_2 += [AffineCouplingLayer(num_features    = len(inference_parameters),
                                                    num_identity    = len(inference_parameters)//2, 
                                                    num_transformed = len(inference_parameters)//2)]
            
        coupling_transform_2 = CouplingTransform(coupling_layers_2)

        #FLOW --------------------------------------------------------------------------------------
        flow_2 = Flow(base_distribution = base_2, 
                    transformation    = coupling_transform_2,
                    embedding_network = kenn_2,
                    prior_metadata    = test_dataset._prior_metadata,
                    ).float().to(device)
        
            #load trained weights
        checkpoint_dir = os.path.join('training_results', 'trained_KENN_3BBH_100epoch')
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_filepath = os.path.join(checkpoint_dir, 'trained_KENN.pt')
        state_dict = torch.load(checkpoint_filepath, map_location=torch.device('cpu'))
        flow_2.load_state_dict(state_dict['model_state_dict'])

         #setup sampler
        sampler_2 = PosteriorSampler(
            flow       = flow_2,
            device     = device,
            output_dir = checkpoint_filepath[:-16])
    
        num_samples=10000

        posterior = sampler.sample_posterior(strain=strain, asd = asd, num_samples=num_samples, restrict_to_bounds = True)
        
        #compare sampled parameters to true parameters
        true_parameters = sampler.flow._post_process_samples(parameters, restrict_to_bounds=False)
        true_parameters = TensorDict.from_dict(true_parameters)
        true_parameters = {key: true_parameters[key].cpu().item() for key in true_parameters.keys()}
 
        print('Sampled parameters vs true parameters medians')
        for par in true_parameters:
            print(f"{par}: {posterior[par].cpu().median():.2f} vs {true_parameters[par]:.2f}")
        
        #generate corner plot
        sampler.plot_corner(injection_parameters=true_parameters)

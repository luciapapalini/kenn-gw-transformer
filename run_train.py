import os
import yaml
import torch
import seaborn as sns
sns.set_theme()
sns.set_context("talk")

from optparse import OptionParser

from deepfastet.config import CONF_DIR
from deepfastet.dataset import DatasetGenerator
from deepfastet.model.flow_utils import get_inference_parameters

from hyperion.core.flow import build_flow
from hyperion.training import (get_LR_scheduler,
                               get_optimizer, 
                               Trainer)
try:
    from hyperion.core.utilities import GWLogger
except: #new hyperion implementation
    from hyperion.core.utilities import HYPERION_Logger as GWLogger


log = GWLogger()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--preload_trained", default=False, action="store_true", help="Load a pretrained model in training_results/KENN directory.")
    parser.add_option("-n", "--name", default=None, help="Name of the model to be trained.")
    parser.add_option("-d", "--device", default= 'cuda', help="Device to use for training. Default is cuda. Can be 'cpu', 'cuda', 'cuda:1'.")
    (options, args) = parser.parse_args()
    
    PRELOAD    = options.preload_trained        
    MODEL_NAME = options.name
    device     = options.device

    if PRELOAD:
        assert MODEL_NAME is not None, 'Please provide the name of the model to be loaded.'
        conf_dir = os.path.join('training_results', MODEL_NAME)
    else:
        conf_dir = CONF_DIR

    conf_yaml  = conf_dir + '/kenn_config.yml'
    PRIOR_PATH = conf_dir + '/population_priors.yml'
    
    with open(conf_yaml, 'r') as yaml_file:
        conf = yaml.safe_load(yaml_file)
        train_conf = conf['training_options']

    #if PRELOAD, load the history file to get the learning rates
    if PRELOAD:
        log.info(f'Loading pretrained model from {conf_dir} directory...')
        import numpy as np
        history_file = os.path.join(conf_dir, 'history.txt')
        _, _, learning_rates = np.loadtxt(history_file, delimiter=',', unpack=True)
        preload_lr = learning_rates[-1] if len(learning_rates) > 0 else learning_rates
            
    #setup checkpoint directory
    
    if MODEL_NAME is None:
        MODEL_NAME = f"KENN_{conf['num_overlapping_signals']['BBH']}_BBH_{conf['duration']}s" #FIXME - make it general
    
    
    checkpoint_dir = os.path.join('training_results', MODEL_NAME)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        log.info(f'Creating directory {checkpoint_dir} to save the model checkpoints...')
    checkpoint_filepath = os.path.join(checkpoint_dir, f'{MODEL_NAME}.pt')
    
    if not PRELOAD:
        #write configuaration file to checkpoint directory
        conf_yaml_write = os.path.join(checkpoint_dir, 'kenn_config.yml')
        with open(conf_yaml_write, 'w') as yaml_file:
            yaml.dump(conf, yaml_file)
        
        #write prior file to checkpoint directory
        conf_prior_write = os.path.join(checkpoint_dir, 'population_priors.yml')
        with open(conf_prior_write, 'w') as yaml_file:
            with open(PRIOR_PATH, 'r') as prior:
                prior = yaml.safe_load(prior)
            yaml.dump(prior, yaml_file)
    
    
    #set device
    if not torch.cuda.is_available():
        device = 'cpu'
    with torch.device(device):

        # inference parameters flags
        inference_parameters_flags = conf['inference_parameters_flags']

        # number of overlapping signals
        num_overlapping_signals = conf['num_overlapping_signals']

        # inference parameters
        inference_parameters = get_inference_parameters(inference_parameters_flags, num_overlapping_signals)
        
        
        #setup Dataset & Dataloader
        log.info('Using device {}'.format(device))
        log.info('Batch size {}'.format(train_conf['batch_size']))      
        # dataset generator arguments
        dataset_kwargs = {
            'prior_filepath'      : PRIOR_PATH,
            'signal_duration'     : conf['duration'],
            'batch_size'          : train_conf['batch_size'],
            'source_kind'         : conf['source_kind'],
            'fixed_signals'       : conf['fixed_signals'],
            'num_fixed_signals'   : num_overlapping_signals['BBH'],  #TODO change and make it general
            'device'              : device,
            'inference_parameters': inference_parameters,
            'add_noise'           : train_conf['add_noise'],
            'npool'               : eval(train_conf['npool'])
                        }
        
        train_dataset = DatasetGenerator(mode='training', **dataset_kwargs)
        val_dataset   = DatasetGenerator(mode='validation', **dataset_kwargs)
                    
    
        #setup KENN & FLOW model ================================================
        kenn_config = {
                    'samples':conf['duration']*train_dataset.fs,
                    'duration_in_s' : conf['duration'],  # s
                    'sampling_rate':train_dataset.fs,  # Hz
                    'n_channels':3,
        }
        
        if not PRELOAD:
            prior_metadata = train_dataset._prior_metadata
            base_distribution_kwargs = conf['base_distribution']
            base_distribution_kwargs['kwargs'] = {'dim': len(inference_parameters)}
            
            flow_kwargs    = conf["flow"] #{'num_coupling_layers': len(inference_parameters) * 3}
            coupling_layers_kwargs = {'num_features'   : len(inference_parameters), 
                                      'num_identity'   : len(inference_parameters)-len(inference_parameters)//2,
                                      'num_transformed': len(inference_parameters)//2}
            
            embedding_network_kwargs = {'model': 'KENN', 'kwargs': kenn_config}

                
            flow = build_flow(prior_metadata           = prior_metadata,
                              flow_kwargs              = flow_kwargs,
                              coupling_layers_kwargs   = coupling_layers_kwargs,
                              base_distribution_kwargs = base_distribution_kwargs,
                              embedding_network_kwargs = embedding_network_kwargs,
                            ).to(device)
            print(flow)
        else:
            flow = build_flow(checkpoint_path=checkpoint_filepath).to(device)   

            
        #set up Optimizer and Learning rate schedulers
        optim_kwargs = {'params': [p for p in flow.parameters() if p.requires_grad], 
                        'lr': train_conf['initial_learning_rate']}
        optimizer = get_optimizer(name='Adam', kwargs=optim_kwargs)

        scheduler_kwargs = train_conf['lr_schedule']['kwargs']
        scheduler_kwargs.update({'optimizer':optimizer})
        scheduler = get_LR_scheduler(name = train_conf['lr_schedule']["scheduler"], 
                                        kwargs = scheduler_kwargs )
        
        
        #set up Trainer
        trainer_kwargs = {
            'flow'               : flow,
            'training_dataset'   : train_dataset,
            'validation_dataset' : val_dataset,
            'optimizer'          : optimizer,
            'scheduler'          : scheduler,
            'device'             : device,
            'checkpoint_filepath': checkpoint_filepath,
            'steps_per_epoch'    : train_conf['steps_per_epoch'],
            'val_steps_per_epoch': train_conf['val_steps_per_epoch'],
            'verbose'            : train_conf['verbose'],
            'add_noise'          : train_conf['add_noise'],
            
            }
        
        kenn_trainer = Trainer(**trainer_kwargs)
            
        kenn_trainer.train(train_conf['num_epochs'], overwrite_history=False if PRELOAD else True)

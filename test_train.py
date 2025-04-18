import os
import yaml
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


from optparse import OptionParser
from tensordict import TensorDict

from deepfastet.dataset import DatasetGenerator
from deepfastet.model.flow_utils import get_inference_parameters

try:
    from hyperion.core import PosteriorSampler, GWLogger
except:
    from hyperion import PosteriorSampler
    from hyperion.core.utilities import HYPERION_Logger as GWLogger

log = GWLogger()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--num_posterior_samples", default=100_000, help="Number of posterior samples to draw")
    parser.add_option("-v", "--verbosity", default=False, action="store_true", help="Verbosity of the flow sampler. (Default: False)")
    parser.add_option("-n", "--name", default=None, help="Name of the model to be tested.")
    parser.add_option("-d", "--device", default= 'cuda:1', help="Device to use for training. Default is cuda. Can be 'cpu', 'cuda', 'cuda:1'.")
    parser.add_option("-p", "--plot", default=True, help="Plot simulated strain.")
    
    
    (options, args) = parser.parse_args()
    
    NUM_SAMPLES    = int(options.num_posterior_samples)
    VERBOSITY      = options.verbosity
    MODEL_NAME     = options.name
    device         = options.device
    PLOT           = options.plot
    
    #Setup & load model --------------------------------------------------
    conf_yaml = f"training_results/{MODEL_NAME}/kenn_config.yml"
    
    with open(conf_yaml, 'r') as yaml_file:
        conf = yaml.safe_load(yaml_file)
        train_conf = conf['training_options']

    PRIOR_PATH = f"training_results/{MODEL_NAME}/population_priors.yml"
    DURATION   = conf['duration']

    log.info('Using device {}'.format(device))
        
    with torch.device(device):
        
        # number of overlapping signals
        num_overlapping_signals = conf['num_overlapping_signals']
        
        # inference parameters flags
        inference_parameters_flags = conf['inference_parameters_flags']
        
        # inference parameters
        inference_parameters = get_inference_parameters(inference_parameters_flags, num_overlapping_signals)
        
        
        #Setup dataset generator
        dataset_kwargs = {
            'prior_filepath'      : PRIOR_PATH,
            'signal_duration'     : conf['duration'],
            'batch_size'          : 1,
            'source_kind'         : conf['source_kind'],
            'fixed_signals'       : conf['fixed_signals'],
            'num_fixed_signals'   : num_overlapping_signals['BBH'],  #TODO change and make it general
            'device'              : device,
            'inference_parameters': inference_parameters,
            'add_noise'           : train_conf['add_noise'],
            'plot_template'       : False,
            'plot_signal'         : False,
            'npool'               : eval(train_conf['npool'])
                        }
        
        test_dataset = DatasetGenerator(mode='test', **dataset_kwargs)
        test_dataset.preload_waveforms()
          

        #SAMPLING --------                
        parameters, whitened_strain, projected_i, asd, snr, tcoals, durations = test_dataset.__getitem__(add_noise=conf['training_options']['add_noise'])
        print(projected_i[0]["E1"][0])
        snr = snr.view(-1)
        print(f"SNR: {snr}")
        tensor_values = [v for d in tcoals for v in d.values()]
        tcoals = torch.unique(torch.cat(tensor_values))
        durations = torch.cat(durations)

        #print(f"Parameters: {parameters}")
        #print(f"Whitened Strain shape: {whitened_strain.shape}")
        
        #print(asd.shape)
        #STRAIN PLOT
        # plt.figure(figsize=(20, 15))
        # t = torch.arange(0, DURATION, 1/conf['fs']) - DURATION/2
        # for i, det in enumerate(test_dataset.det_names):
        #     plt.subplot(3, 1, i+1)
        #     plt.plot(whitened_strain[0][i].cpu().numpy())
        #     plt.title(det)           
        # plt.savefig(f'training_results/{MODEL_NAME}/strain.png', dpi=200)
        #plt.show()

        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "Computer Modern"
        plt.rcParams["font.size"]   = 18
        # template plot
        times = np.arange(-test_dataset.signal_duration, 0, 1/test_dataset.fs)
        colors = ["darkblue", "darkred", "darkorange"]
        
        colormap = cm.magma 
        n_curves = 5
        colors = [colormap(i / (n_curves - 1)) for i in range(n_curves)]
      

        
        for det in test_dataset.ET.arms:
            template_sum = np.zeros_like(times)
            
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            for i in range(len(projected_i)):
                template = projected_i[i][det].cpu().numpy()[0]
                template_sum += template
                
                ax[0].plot(times, template, alpha = 1, linestyle = '-', linewidth = 1.1, color=colors[i+1], label=r'BBH$_{'+str(i+1)+'}$ signal')
                ax[0].set_xlim(-15, -5)
                
                ax[0].set_ylabel('Strain')
            ax[1].set_ylabel('Strain')
            ax[1].set_xlabel(r'$t - t_{GPS}$ [s]')  
            ax[1].plot(times, template_sum, color='midnightblue', alpha=1, linestyle = '-', linewidth=1.1, label='Sum of signals')
            ax[1].set_xlim(-15, -5)
            #set minor xticks on
            # set unlabelled xticks
            #remove xticks from ax 0

            ax[0].set_xticklabels([])
            ax[0].minorticks_on()
            ax[1].minorticks_on()
            ax[0].grid(True, which='both', linestyle='-.', linewidth=0.8, alpha=0.2)
            ax[1].grid(True, which='both', linestyle='-.', linewidth=0.8, alpha=0.2)

            ax[0].set_xticks(np.arange(-15, -5, 0.5), minor=True)
            ax[1].set_xticks(np.arange(-15, -5, 0.5), minor=True)
            #ax[0].set_xticks(np.arange(-15, -5, 0.5), minor=True)
                #ax.tick_params(axis='both', labelsize=18)
                #for j in range(n_signals['BBH']):
                #    axs[i].axvline(out_prior_kind[f'tcoal_BBH_{j+1}'].cpu().numpy()[0], color='red')

            # add legend
            ax[0].legend(loc='upper center', fontsize=12)
            ax[1].legend(loc='upper center', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'training_results/{MODEL_NAME}/template_{det}.pdf', dpi=200, bbox_inches='tight')
            plt.show()
            
        


        #set up Sampler
        checkpoint_path = f'training_results/{MODEL_NAME}/{MODEL_NAME}.pt'
        
        sampler = PosteriorSampler(flow_checkpoint_path  = checkpoint_path, 
                                   num_posterior_samples = NUM_SAMPLES,
                                   device                = device)

        
        #for key in sampler.flow.prior_metadata.keys():
        #    print('\n\n')
        #    print(key, test_dataset._prior_metadata[key])
        #    print('\n\n')

        print(sampler.flow)

        
        sampler.flow.prior_metadata = test_dataset._prior_metadata
        posterior = sampler.sample_posterior(strain = whitened_strain, asd = asd, restrict_to_bounds=True, post_process=True)
        
        #compare sampled parameters to true parameters
        true_parameters = sampler.flow._post_process_samples(parameters, restrict_to_bounds=False)
        true_parameters = TensorDict.from_dict(true_parameters)
        true_parameters = {key: true_parameters[key].cpu().item() for key in true_parameters.keys()}

        
        log.info('Sampled parameters medians vs true parameters')
        print('\n')
        print(f"{'Parameter':<15} {'Median':<10} {'Truth'}")
        print(f"{'---------':<15} {'------':<10} {'-----'}")
        for par in true_parameters:
            print(f"{par:<15} {posterior[par].cpu().median():<10.3f} {true_parameters[par]:<10.3f}")
        print('\n')
        #generate corner plot
        sampler.plot_corner(injection_parameters=true_parameters, figname=f'training_results/{MODEL_NAME}/corner_plot.pdf')
        

        
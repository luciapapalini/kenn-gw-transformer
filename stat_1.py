import os
import yaml
import bilby
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
#sns.set_theme()
#sns.set_context("talk")
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from tqdm import tqdm
from optparse import OptionParser
from tensordict import TensorDict

from deepfastet.dataset import DatasetGenerator
from deepfastet.model.flow_utils import get_inference_parameters

from pp_plot import make_pp_plot


try:
    from hyperion.core import PosteriorSampler, GWLogger
except:
    from hyperion import PosteriorSampler
    from hyperion.core.utilities import HYPERION_Logger as GWLogger

log = GWLogger()

from clustering import PosteriorClustering


def normalize(tensor):
    return (tensor - tensor.mean()) / tensor.std()

def cross_correlation(x, y):
    # Normalizza x e y
    x_norm = normalize(x)
    y_norm = normalize(y)

    # Zero-padding per emulare la correlazione incrociata
    n = len(x)
    padding = n - 1

    # Segui lo stesso procedimento di prima
    y_padded_norm = torch.nn.functional.pad(y_norm, (padding, padding))
    correlation_norm = torch.nn.functional.conv1d(
        y_padded_norm.unsqueeze(0).unsqueeze(0),
        x_norm.flip(0).unsqueeze(0).unsqueeze(0),
        padding=0
    ).squeeze()

    lags = torch.arange(-padding, padding + 1)

    # Trova il lag con massima correlazione normalizzata
    max_lag_norm = lags[torch.argmax(correlation_norm)].item()

    return max_lag_norm



def pearson_correlation(x, y):
    """
    Computes the Pearson correlation coefficient between two signals.
    
    Args:
        x (torch.Tensor): 1D tensor representing the first signal.
        y (torch.Tensor): 1D tensor representing the second signal.
    
    Returns:
        float: Pearson correlation coefficient.
    """
    # Ensure x and y are 1D tensors
    x = x.flatten()
    y = y.flatten()
    
    # Compute means
    mean_x = x.mean()
    mean_y = y.mean()
    
    # Compute numerator (covariance)
    numerator = torch.sum((x - mean_x) * (y - mean_y))
    
    # Compute denominator (standard deviations product)
    denominator = torch.sqrt(torch.sum((x - mean_x)**2)) * torch.sqrt(torch.sum((y - mean_y)**2))
    
    # Compute Pearson correlation coefficient
    r = numerator / denominator
    
    return r.item()  



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--num_posterior_samples", default=5_000, help="Number of posterior samples to draw")
    parser.add_option("-v", "--verbosity", default=False, action="store_true", help="Verbosity of the flow sampler. (Default: False)")
    parser.add_option("-n", "--name", default=None, help="Name of the model to be trained.")
    parser.add_option("-d", "--device", default= 'cuda:1', help="Device to use for training. Default is cuda. Can be 'cpu', 'cuda', 'cuda:1'.")
    parser.add_option("-p", "--num_posteriors", default=128, help="Number of posteriors to draw for the PP plot.")
    parser.add_option("-t", "--test_population", default=None, help="Tag of the alternative population to test")
    
    (options, args) = parser.parse_args()
    
    NUM_SAMPLES    = int(options.num_posterior_samples)
    VERBOSITY      = options.verbosity
    MODEL_NAME     = options.name
    device         = options.device
    PLOT           = False
    NUM_POSTERIORS = int(options.num_posteriors)
    TEST_POP_NAME   = options.test_population
    
    #Setup & load model --------------------------------------------------
    conf_yaml = f"training_results/{MODEL_NAME}/kenn_config.yml"
    
    with open(conf_yaml, 'r') as yaml_file:
        conf = yaml.safe_load(yaml_file)
        train_conf = conf['training_options']

    # if TEST_POP_NAME is not None, change the prior path
    PRIOR_PATH = f"training_results/{MODEL_NAME}/population_priors.yml"
    SAVING_PATH = f"training_results/{MODEL_NAME}"
    if TEST_POP_NAME is not None:
        PRIOR_PATH = f"training_results/{MODEL_NAME}/test-pop_{TEST_POP_NAME}/population_priors_{TEST_POP_NAME}.yml"
        SAVING_PATH = f"training_results/{MODEL_NAME}/test-pop_{TEST_POP_NAME}"

    # print info on if we have a testing population and which are the directories
    print(f'Prior path: {PRIOR_PATH}')
    print(f'Saving path: {SAVING_PATH}')

    stats_dir = f"{SAVING_PATH}/stats"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)


    DURATION   = conf['duration']

    log.info('Using device {}'.format(device))
        
    with torch.device(device):
        
        # number of overlapping signals
        num_overlapping_signals = conf['num_overlapping_signals']
        
        # inference parameters flags
        inference_parameters_flags = conf['inference_parameters_flags']
        
        # inference parameters
        inference_parameters = get_inference_parameters(inference_parameters_flags, num_overlapping_signals)
        
        print(f'I am before the dataset generator')
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
            'plot_template'       : PLOT,
            'plot_signal'         : PLOT,
                        }
        
        test_dataset = DatasetGenerator(mode='test', **dataset_kwargs)
        test_dataset.preload_waveforms()
        print(f'I am after the dataset generator')

        #set up Sampler
        checkpoint_path = f'training_results/{MODEL_NAME}/{MODEL_NAME}.pt'
        
        sampler = PosteriorSampler(flow_checkpoint_path  = checkpoint_path, 
                                   num_posterior_samples = NUM_SAMPLES,
                                   device                = device)
        
        cluster = PosteriorClustering(k=1000, verbose=False, cluster_on=["tcoal", "q"])
        
        
        # bilby wants labels for priors, create stupid priors
        fake_priors = {}
        for par in inference_parameters:
            fake_priors[par] = bilby.core.prior.Uniform(0, 1, latex_label=sampler.latex_labels()[par])
        
        posteriors = []
        truths = []
        error = {}
        snr_tot = []
        tcoals_tot = []
        durations_tot = []
        time_overlap = []
        correlation = []
        pearson = [] #old useless?
        pearson_coeff = {}

        # we initialize a dictionary (a new one yay) with lists, to save every couple of pearsons
        x = list(range(num_overlapping_signals['BBH']))
        pairs = list(itertools.combinations(x, 2))

        for det in ['E1', 'E2', 'E3']:
            for pair in pairs:
                pearson_coeff[f'{det}_{pair[0]}_{pair[1]}'] = []

        mode = {}
        median = {}
        true_parameters_values = {}

        log.info(f"Drawing {NUM_POSTERIORS} posteriors...")
        for i in tqdm(range(NUM_POSTERIORS)):
            
            #SAMPLING --------                
            parameters, whitened_strain, projected_templates, asd, snr, tcoals, durations = test_dataset.__getitem__(add_noise=conf['training_options']['add_noise'])

            # adjustments
            snr = snr.view(-1)
            tensor_values = [v for d in tcoals for v in d.values()]
            tcoals = torch.unique(torch.cat(tensor_values))
            durations = torch.cat(durations)

            # append
            snr_tot.append(snr)
            tcoals_tot.append(tcoals)
            durations_tot.append(durations)

            # time overlaps of a single sample

            time_overlap_sample = []

            # time overlap stat
            for k in range(len(tcoals)):
                if k == 0 or k == 1:
                    e1 = tcoals[k]
                    e2 = tcoals[k+1]

                    s1 = tcoals[k]-durations[k]
                    s2 = tcoals[k+1]-durations[k+1]

                    tau = ((min(e1,e2)-max(s1,s2))/((abs(e1-e2)+1)*(max(e1,e2)-min(s1,s2)))).cpu().numpy()
                    time_overlap_sample.append(tau)

                elif k == 2:
                    e1 = tcoals[k]
                    e2 = tcoals[0]

                    s1 = tcoals[k]-durations[k]
                    s2 = tcoals[0]-durations[0]

                    tau = ((min(e1,e2)-max(s1,s2))/((abs(e1-e2)+1)*(max(e1,e2)-min(s1,s2)))).cpu().numpy()
                    time_overlap_sample.append(tau)

                time_overlap.append(time_overlap_sample)

            # correlations 

            templates_oredered = {}

            for det in ['E1', 'E2', 'E3']:
                stack = []
                for k in range(num_overlapping_signals['BBH']):
                    stack.append(projected_templates[k][det][0])
                templates_oredered[det] = torch.stack(stack)

             # total number of point in the time chunk
            tot_points = test_dataset.signal_duration * test_dataset.fs

            correlation_lag = {}

            for det in ['E1', 'E2', 'E3']:
                dummy = 0
                dummy_pearson = 0
                for pair in pairs:
                    # correlation
                    lag = cross_correlation(templates_oredered[det][pair[0]], templates_oredered[det][pair[1]])
                    dummy += (lag/tot_points)**2
                    #pearson
                    pears = pearson_correlation(templates_oredered[det][pair[0]], templates_oredered[det][pair[1]])
                    pearson_coeff[f'{det}_{pair[0]}_{pair[1]}'].append(pears)
                    dummy_pearson += pears**2

                correlation_lag[det] = np.sqrt(dummy)/np.sqrt(num_overlapping_signals['BBH'])
                #pearson_coeff[det] = np.sqrt(dummy_pearson)/np.sqrt(num_overlapping_signals['BBH'])
                

            correlation.append(np.sqrt(sum([correlation_lag[det]**2 for det in correlation_lag]))/np.sqrt(3))
            #pearson.append(np.sqrt(sum([pearson_coeff[det]**2 for det in pearson_coeff]))/np.sqrt(3))
            
 
            #print(f'correlation lag {correlation_lag}')
            
            #sample the posterior
            posterior = sampler.sample_posterior(strain=whitened_strain, asd=asd, post_process=True, restrict_to_bounds=True, verbose=VERBOSITY)

            #relabel the bbhs with clustering algorithm
            try:
                _, posterior = cluster(posterior)
            except Exception as e:
                print(f"Error in clustering: {e}")
                pass
            
            #compare sampled parameters to true parameters
            true_parameters = sampler.flow._post_process_samples(parameters, restrict_to_bounds=False)
            true_parameters = TensorDict.from_dict(true_parameters)
            true_parameters = {key: true_parameters[key].cpu().item() for key in true_parameters.keys()}

            truths.append(true_parameters)

            # save true parameters values
            if i == 0:
                true_parameters_values = true_parameters
            else:
                for key in true_parameters.keys():
                    true_parameters_values[key] = np.append(true_parameters_values[key], true_parameters[key])

            #posterior_bilby = sampler.to_bilby(posterior, injection_parameters=true_parameters, priors=fake_priors)
            #posteriors.append(posterior_bilby)
            posteriors.append(posterior.cpu().to_dict())

            # print posterior keys
            #print(posterior)
            #print(f'Posterior keys: {list(posterior.keys())}')

            for key in list(posterior.keys()):
                if any(pname in key for pname in ['M', 'Mchirp', 'q', 'tcoal']):
                    #if 'chirp' in key:
                    #    continue
                    median_val = posterior[key].median().cpu().numpy()
                    
                    #determine the mode of the posterior through KDE gaussian kernels
                    # samples = posterior[key].cpu().numpy()
                    # kde = gaussian_kde(samples)
                    # x = np.linspace(samples.min(), samples.max(), 1000)
                    # kde_values = kde(x)
                    # mode_val = x[np.argmax(kde_values)]
                    mode_val = median_val
                    
                    #compute the relative error
                    truth = true_parameters[key]
                    delta = (median_val - truth) / truth
                    
                    if i == 0:
                        error[key] = delta
                        mode[key] = mode_val
                        median[key] = median_val
                    else:
                        error[key] = np.append(error[key], delta)
                        mode[key] = np.append(mode[key], mode_val)
                        median[key] = np.append(median[key], median_val)


            if VERBOSITY:
                print('\n')
                print(f"{'Parameter':<15} {'Median':<10} {'Truth'}")
                print(f"{'---------':<15} {'------':<10} {'-----'}")
                for par in true_parameters:
                    print(f"{par:<15} {posterior[par].cpu().median_val():<10.3f} {true_parameters[par]:<10.3f}")
                print('\n')
                
    #renaming the keys of the error dictionary
    error_keys = list(error.keys())
    '''
    for parameter in error_keys:
        print(parameter)
        par_name, kind, number = parameter.split('_')
        
        if par_name == 'tcoal':
            new_key = f'$t_{number}$'
        elif par_name == 'q':
            new_key = f'$q_{number}$'
        elif par_name == 'M':
            new_key = f'$M_{number}$ [M$_\odot$]'
        elif par_name == 'Mchirp':
            new_key = rf'$\mathcal{{M}}_{number}$ [M$_\odot$]'
            
        
        error[new_key] = error.pop(parameter)            
'''
    # pp plot with bilby
    log.info('Generating PP plot...')
    #bilby.core.result.make_pp_plot(posteriors, filename=f"{SAVING_PATH}/pp_plot.png")
    make_pp_plot(posteriors, truths, out_dir=SAVING_PATH )

    
    # violin plot with error with seaborn
    # log.info('Generating violin plot...')
    # palette = sns.color_palette("coolwarm", as_cmap=False)
    
    # plt.rcParams.update({'font.size': 20})
    # plt.figure(figsize=(12, 12))
    error_df = pd.DataFrame(error)
    # sns.violinplot(data=error_df, palette=palette)
    # plt.ylabel('$\delta p / p$')
    # plt.xticks(rotation=30)
    # plt.minorticks_on()
    # plt.savefig(f"{SAVING_PATH}/violin_plot.png", dpi=300, bbox_inches='tight')
    

    # overlap stat
    print(f'timeoverlap: {time_overlap}')
    print(f'to type {type(time_overlap)}')

    '''
    plt.figure(figsize=(12, 12))
    for e_key in error_keys:
        print(time_overlap, error[e_key])
        plt.plot(time_overlap, error[e_key], 'o', label=e_key)
    plt.xlabel('Time overlap')
    plt.ylabel('Relative error')
    plt.minorticks_on()
    plt.savefig(f"training_results/{MODEL_NAME}/time_overlap.png", dpi=300, bbox_inches='tight')
    '''

    # file to save in different columns all the needed stats, snr, tcoals, durations, time_overlap
    # save them as a pandas dataframe
    # save the dataframe as a csv file
    # we need to unpack the dictionaries in snr, tcoals, durations
    multiple_stats = ['snr', 'tcoals', 'durations', 'time_overlap']
    df_cols = []
    for stat in multiple_stats:
        for num in range(num_overlapping_signals['BBH']):
            df_cols.append(f'{stat}_{num}')

    df_cols.append('correlation')
    #df_cols.append('pearson')

    df = pd.DataFrame(columns=df_cols)
    
    # now we need to assign the values to the dataframe
    for sample in range(len(snr_tot)):
        for i in range(num_overlapping_signals['BBH']):
            print(f'value of snr_{i}: {snr_tot[sample][i]}')
            df.loc[sample, f'snr_{i}']= snr_tot[sample][i].cpu().numpy()
            df.loc[sample, f'tcoals_{i}'] = tcoals_tot[sample][i].cpu().numpy()
            df.loc[sample, f'durations_{i}'] = durations_tot[sample][i].cpu().numpy()
            df.loc[sample, f'time_overlap_{i}'] = time_overlap[sample][i]
        df.loc[sample, 'correlation'] = correlation[sample]
        #df.loc[sample, 'pearson'] = pearson[sample]

    print(df)

    df.to_csv(f"{SAVING_PATH}/stats/stats_withpearson.csv", index=False)


    # save errors on another csv file
    error_df.to_csv(f"{SAVING_PATH}/stats/errors_withpearson.csv", index=False)


    # turn mode and median and truth values into a pandas dataframe and save them in a single file called mode_median_truth.csv
    mode_df = pd.DataFrame.from_dict(mode)
    median_df = pd.DataFrame.from_dict(median)
    truth_df = pd.DataFrame.from_dict(true_parameters_values, orient='index').T

    #mode_median_truth = pd.concat([mode_df, median_df, truth_df], axis=0)
    mode_df.to_csv(f"{SAVING_PATH}/stats/mode.csv", index=False)
    median_df.to_csv(f"{SAVING_PATH}/stats/median.csv", index=False)
    truth_df.to_csv(f"{SAVING_PATH}/stats/truth.csv", index=False)

    # pearson coefficients

    pearson_df = pd.DataFrame.from_dict(pearson_coeff)
    pearson_df.to_csv(f"{SAVING_PATH}/stats/pearson_coefficients_pairs.csv", index=False)

    

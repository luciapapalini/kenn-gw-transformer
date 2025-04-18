import os
import yaml
import bilby
import torch
import numpy as np
import pandas as pd
import seaborn as sns
#sns.set_theme()
#sns.set_context("talk")
import matplotlib.pyplot as plt
import itertools
import re


from scipy.stats import gaussian_kde

from tqdm import tqdm
from optparse import OptionParser
from tensordict import TensorDict

from pastamarkers import markers



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-n", "--name", default=None, help="Name of the model to be trained.")
    parser.add_option("-t", "--test_population", default=None, help="Tag of the alternative population to test")

    
    (options, args) = parser.parse_args()

    MODEL_NAME     = options.name
    TEST_POP_NAME  = options.test_population

    # if TEST_POP_NAME is not None, change the prior path
    PRIOR_PATH = f"training_results/{MODEL_NAME}/population_priors.yml"
    SAVING_PATH = f"training_results/{MODEL_NAME}"
    MAIN_PATH = f"training_results/{MODEL_NAME}"
    if TEST_POP_NAME is not None:
        PRIOR_PATH = f"training_results/{MODEL_NAME}/test-pop_{TEST_POP_NAME}/population_priors_{TEST_POP_NAME}.yml"
        SAVING_PATH = f"training_results/{MODEL_NAME}/test-pop_{TEST_POP_NAME}"

    pearson_dir = f"{SAVING_PATH}/pearson_coefficient"
    if not os.path.exists(pearson_dir):
        os.makedirs(pearson_dir)


    #Setup & load model --------------------------------------------------
    conf_yaml = f"training_results/{MODEL_NAME}/kenn_config.yml"

    with open(conf_yaml, 'r') as yaml_file:
        conf = yaml.safe_load(yaml_file)
        train_conf = conf['training_options']

    # number of overlapping signals
    num_overlapping_signals = conf['num_overlapping_signals']

    # parameters
    parameters = ['M_BBH', 'q_BBH', 'Mchirp_BBH', 'tcoal_BBH']

    # open csv error and stat
    error = pd.read_csv(f"{SAVING_PATH}/stats2/errors_withpearson.csv")
    stat = pd.read_csv(f"{SAVING_PATH}/stats2/stats_withpearson.csv")
    mode = pd.read_csv(f"{SAVING_PATH}/stats2/mode.csv")
    median = pd.read_csv(f"{SAVING_PATH}/stats2/median.csv")
    truth = pd.read_csv(f"{SAVING_PATH}/stats2/truth.csv")
    pearson_pairs = pd.read_csv(f"{SAVING_PATH}/stats2/pearson_coefficients_1vs2.csv")

    # error with median
    error_median = (median - truth)/truth  # TODO NON GLI PIACE ABS
    error_mode = (mode - truth)/truth

    # ZA WARUDO
    colors = ['#ffa62b', '#49A3C9', '#9E5793']

    #_________________PEARSON COEFFICIENTS CALCULATION
    
    pears = []
    
    for index in range(num_overlapping_signals['BBH']):
        detector_sum = 0
        for det in ['E1', 'E2', 'E3']:
            detector_sum += pearson_pairs[f'{det}_{index}']**2
        pears.append(np.sqrt(detector_sum/3))

    # quadrature sum between the 3 signals
    pearson_statistic = np.sqrt(np.sum(np.array(pears)**2, axis=0)/3)

    # pears = np.array(pears).T
    # pears_mean = np.mean(pears, axis=1)
    # print(f"Pearson mean: {pears_mean.shape}")
    # print(f"Pearson: {pears.shape}")
    # pearson_statistic = []
    # for p, pmean in tqdm(zip(pears, pears_mean), total=len(pears)):
    #     stat = len(p[p>0.05])
    #     pearson_statistic.append(stat)
    # pearson_statistic = np.array(pearson_statistic)

    # p_unique, p_counts = np.unique(pearson_statistic, return_counts=True)
    # print(f"Unique pearson: {p_unique}")
    # print(f"Counts pearson: {p_counts}")
    # #print(f"Pearson statistic: {pearson_statistic}")

    # histogram of the pearson coefficients
    fig, ax = plt.subplots()
    ax.hist(pearson_statistic, 100)
    ax.set_yscale('log')
    ax.set_xlabel("Pearson Coefficient")
    plt.savefig(f"{SAVING_PATH}/pearson_coefficient/pearson_hist.png")

    #_________________VIOLIN PLOT ERRORS GLOB
    palette = sns.color_palette("coolwarm", as_cmap=False)
    
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(16, 10))
    error_df = error.iloc[:, :-4].where(error.iloc[:, :-4] <= 1)
    sns.violinplot(data=error_df, palette=palette)
    plt.ylabel('$\delta p / p$')
    plt.xticks(rotation=30)
    plt.minorticks_on()
    plt.savefig(f"{SAVING_PATH}/violin_plot_2.png", dpi=300, bbox_inches='tight')



    #_________________________PEARSON
    # Calculate Pearson coefficient related variables
    for parameter in parameters:
        error_4pearson = error_mode

        error_4pearson[f'pearson'] = pearson_statistic

        mask = (error_4pearson[[f'{parameter}_1', f'{parameter}_2', f'{parameter}_3']] < 1).all(axis=1) & (error_4pearson[f'pearson']<4)


        # Apply the mask to get filtered data
        filtered_error_4pearson = error_4pearson[mask]

        # Violin plot with filtered errors
        num_bins = 4
        #binned = pd.qcut(filtered_error_4pearson[f'pearson'], q=num_bins, duplicates='drop')

        binned = pd.cut(filtered_error_4pearson[f'pearson'], num_bins)
        # print number of samples in each bin
        print('\n pearson\n', filtered_error_4pearson.groupby(binned, observed=True)[f'pearson'].count())

        fig, ax = plt.subplots(figsize=(12, 8))  # Improved figure dimensions
    
        #colors = ['#ffa62b', '#49A3C9', '#9E5793']
        colors = ['#fd9674', '#c96497', '#734094'] # plasma 

        for idx, which_bbh in enumerate(range(3, 0, -1)):

            filtered_errors = filtered_error_4pearson[f'{parameter}_{which_bbh}']#[0:1000]

            #fig2, ax2 = plt.subplots(figsize=(10, 6))
            #ax2.hist(filtered_errors, bins=100, alpha=0.5, color=colors[idx], label=f'{parameter}_{which_bbh}', density=True)
            #fig2.savefig(f"{SAVING_PATH}/pearson_coefficient/hist_{parameter}_{which_bbh}_range{idx}.png")

            # Use filtered binned and errors for violin plot
            sns.violinplot(
                x=binned,
                y=filtered_errors,
                ax=ax,
                alpha=0.8,
                color=colors[idx],
                fill=True,
                linecolor=colors[idx],
                inner=None,
                label=f'{parameter}_{which_bbh}'
            )

        # Remove duplicate labels from the legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="BBH Components", loc="upper right")

        # Set axis labels and layout adjustments
        ax.set_xlabel("Pearson Coefficient", fontsize=14)
        ax.set_ylabel(r"$\delta{p}/p$", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{SAVING_PATH}/pearson_coefficient/pearson_violin_{parameter}_mode.png")
        plt.show()


    # Plot the distributions of snrs of the 3 BBHs
    fig, ax = plt.subplots(figsize=(10, 6))

    for which_bbh in range(1, 4):
        sns.histplot(stat[f'snr_{which_bbh-1}'], kde=True, ax=ax, label=f"BBH{which_bbh}")
        
    ax.set_xlabel("SNR")
    ax.set_ylabel("Density")
    ax.set_title("SNR Distributions of BBHs")
    ax.legend()
    plt.savefig(f"{SAVING_PATH}/snr_distributions.png")



    # Plot error on Mchirp vs error on q for the 3 BBHs
    fig, ax = plt.subplots(figsize=(10, 6))

    for which_bbh in range(1, 2):
        ax.scatter(error[f'Mchirp_BBH_{which_bbh}'], error[f'q_BBH_{which_bbh}'], label=f"BBH{which_bbh}")

    # draw diagonal
    ax.plot([-1, 4], [-1, 4], color='r', linestyle='--')

    ax.set_xlabel("Error on Mchirp")
    ax.set_ylabel("Error on q")
    ax.set_title("Error on Mchirp vs Error on q")
    ax.legend()
    plt.savefig(f"{SAVING_PATH}/error_on_mchirp_vs_q.png")

    
    # print length of mode, median and truth
    #print(len(mode), len(median), len(truth))


    # plot (mode-truth) vs mode for a parameter we fix 
    parameter = 'Mchirp_BBH'
    fig, ax = plt.subplots(figsize=(10, 6))

    for which_bbh in range(1, 4):
        ax.scatter(mode[f'{parameter}_{which_bbh}'], mode[f'{parameter}_{which_bbh}'] - truth[f'{parameter}_{which_bbh}'], label=f"BBH{which_bbh}")

    ax.set_xlabel("Mode")
    ax.set_ylabel("Mode - Truth")
    ax.set_title("Mode vs Mode - Truth")
    ax.legend()
    plt.savefig(f"{SAVING_PATH}/mode_vs_mode_minus_truth.png")



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
    parameters = ['M_BBH', 'Mchirp_BBH', 'q_BBH', 'tcoal_BBH']

    # open csv error and stat
    error = pd.read_csv(f"{SAVING_PATH}/stats/errors_withpearson.csv")
    stat = pd.read_csv(f"{SAVING_PATH}/stats/stats_withpearson.csv")
    mode = pd.read_csv(f"{SAVING_PATH}/stats/mode.csv")
    median = pd.read_csv(f"{SAVING_PATH}/stats/median.csv")
    truth = pd.read_csv(f"{SAVING_PATH}/stats/truth.csv")
    pearson_pairs = pd.read_csv(f"{SAVING_PATH}/stats/pearson_coefficients_1vs2.csv")

    # error with median
    error_median = (median - truth)/truth  # TODO NON GLI PIACE ABS
    error_mode = (mode - truth)/truth

    # ZA WARUDO
    colors = ['#ffa62b', '#49A3C9', '#9E5793']


    #which_bbh = 2

    #_________________PEARSON COEFFICIENTS CALCULATION
    x = list(range(num_overlapping_signals['BBH']))
    pairs = list(itertools.combinations(x, 2))

    pears = []
    
    # Compile a regex pattern to extract the pair part (e.g., "0_1") from each key.
    pattern = re.compile(r'^[^_]+_(\d+_\d+)$')

    # Group keys by the pair identifier.
    grouped_keys = {}
    for key in pearson_pairs:
        match = pattern.match(key)
        if match:
            pair_id = match.group(1)
            grouped_keys.setdefault(pair_id, []).append(key)

    # For each unique pair, sum all the numbers from the lists of the corresponding keys.
    pair_sums = {}
    for pair_id, keys in grouped_keys.items():
        total = 0
        for key in keys:
            total += (pearson_pairs[key].values)**2
        pair_sums[pair_id] = np.sqrt(total/3)


    # Now, pair_sums holds the sum for each pair identifier.
    # You can iterate over it or use it further in your code.
    # for pair_id, total in pair_sums.items():
    #     print(f"Pair {pair_id}: Sum = {total}")

    # quadrature sum of the 3 pair sums
    pears = []
    
    for index in range(num_overlapping_signals['BBH']):
        detector_sum = 0
        for det in ['E1', 'E2', 'E3']:
            detector_sum += pearson_pairs[f'{det}_{index}']**2
        pears.append(np.sqrt(detector_sum/3))

    # quadrature sum between the 3 signals
    #pearson_statistic = np.max(np.array(pears), axis=0)#np.sqrt(np.sum(np.array(pears)**2, axis=0)/3)

    pears = np.array(pears).T
    pears_mean = np.mean(pears, axis=1)
    print(f"Pearson mean: {pears_mean.shape}")
    print(f"Pearson: {pears.shape}")
    pearson_statistic = []
    for p, pmean in tqdm(zip(pears, pears_mean), total=len(pears)):
        stat = len(p[p>0.05])
        pearson_statistic.append(stat)
    pearson_statistic = np.array(pearson_statistic)
    #print(f"Pearson statistic: {pearson_statistic}")

    # histogram of the pearson coefficients
    fig, ax = plt.subplots()
    ax.hist(pearson_statistic, 100)
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
    
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    axs = axs.flatten()  # Flatten for easier iteration

    #colors = ['#ffa62b', '#49A3C9', '#9E5793']
    #colors = ['#49A3C9', '#ffa62b', '#9E5793']
    #colors = ['#fd9674', '#c96497', '#734094'] # plasma 
    #colors = ['#ffa62b', '#c96497', '#9E5793'] # plasma 
    colors = ['#ffa62b', '#c96497', '#734095']

    labels_param = [r'M', r'\mathcal{M}', r'q', r't_{\mathrm{merger}}']
    lab = 0

    # Loop over each parameter and corresponding subplot axis
    for iax, (ax, parameter) in enumerate(zip(axs, parameters)):
        # Prepare data for the current parameter
        error_4pearson = error_mode.copy()  # Use a copy if modifying the DataFrame
        error_4pearson['pearson'] = pearson_statistic

        # Create a mask to filter the data
        mask = (
            (error_4pearson[[f'{parameter}_1', f'{parameter}_2', f'{parameter}_3']] < 0.9).all(axis=1) &
            (error_4pearson['pearson'] < 4)
        )
        filtered_error_4pearson = error_4pearson[mask]

        # Bin the 'pearson' data into three bins
        num_bins = 4
        binned = pd.cut(filtered_error_4pearson['pearson'], num_bins)
        print(f'\n Pearson counts for {parameter} \n', 
            filtered_error_4pearson.groupby(binned, observed=True)['pearson'].count())
        
        ax.axhline(0, color='black', lw=1, ls='--', alpha=0.5)

        # Loop to create a violin plot for each of the three BBH components
        for idx, which_bbh in enumerate(range(3, 0, -1)):
            filtered_errors = filtered_error_4pearson[f'{parameter}_{which_bbh}']

            sns.violinplot(
                x=binned,
                y=filtered_errors,
                ax=ax,
                alpha=0.75,
                color=colors[idx],
                fill=True,
                linecolor=colors[idx],
                inner=None,
                #bw_adjust=2,
                bw_method='scott',
                linewidth=2,
                density_norm='area',
                common_norm=True,
            )
        ax.set_xticklabels(np.arange(num_bins))
        # Set axis labels for each subplot
        ax.set_title(f"${labels_param[lab]}$", fontsize=25, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.6'))
        if iax in [2, 3]:
            ax.set_xlabel(r"$\mathcal{C}$", fontsize=30)
        else:
            ax.set_xlabel("")
        if iax in [0, 2]:
            #ax.set_ylabel(rf"$\delta {labels_param[lab]}\,/\,{labels_param[lab]}$", fontsize=20)
            ax.set_ylabel(r"$\dfrac{\delta p}{p_{\rm true}}$", fontsize=30)
        else:
            ax.set_ylabel("")
       
        lab += 1

    # Create a global legend placed above the subplots in a horizontal layout
    legend_elements = [
        Line2D([0], [0], color=colors[2], lw=14, label='BBH 1'),
        Line2D([0], [0], color=colors[1], lw=14, label='BBH 2'),
        Line2D([0], [0], color=colors[0], lw=14, label='BBH 3')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, frameon=False, fontsize=28)

    # Set all y-axes to use the same ticks and limits
    y_ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    for ax in axs:
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks(y_ticks)

    # Adjust layout and save the combined figure
    #plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f"{SAVING_PATH}/pearson_coefficient/pearson_violin_combined_mode.pdf", dpi=300)
    plt.show()
    #_________________________SNR DISTRIBUTIONS
    # Plot the distributions of snrs of the 3 BBHs
    fig, ax = plt.subplots(figsize=(10, 6))

    for which_bbh in range(1, 4):
        sns.histplot(stat[f'snr_{which_bbh-1}'], kde=True, ax=ax, label=f"BBH{which_bbh}")
        
    ax.set_xlabel("SNR")
    ax.set_ylabel("Density")
    ax.set_title("SNR Distributions of BBHs")
    ax.legend()
    plt.savefig(f"{SAVING_PATH}/snr_distributions.png")

    # from matplotlib.lines import Line2D

    # plt.rcParams.update({'font.size': 20})

    # # Create a figure with 4 subplots arranged in a 2x2 grid
    # fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    # axs = axs.flatten()  # Flatten for easier iteration

    # # Colors for the three BBH components.
    # # (Note: In our plotting loop, we loop over BBH in the order 3, 2, 1.)
    # colors = ['#ffa62b', '#49A3C9', '#9E5793']
    # # Labels for y-axis per parameter
    # labels_param = [r'M', r'\mathcal{M}', r'q', r't_{\mathrm{merger}}']
    # lab = 0

    # # Offsets to separate violins within each bin
    # offsets = [-0.25, 0, 0.25]  # one offset per BBH component

    # # Loop over each parameter and corresponding subplot axis
    # for ax, parameter in zip(axs, parameters):
    #     # Prepare the data for this parameter
    #     error_4pearson = error_mode.copy()
    #     error_4pearson['pearson'] = pearson_statistic

    #     # Create a mask to filter the data
    #     mask = (
    #         (error_4pearson[[f'{parameter}_1', f'{parameter}_2', f'{parameter}_3']] < 1).all(axis=1) &
    #         (error_4pearson['pearson'] < 0.5)
    #     )
    #     filtered_error_4pearson = error_4pearson[mask]

    #     # Bin the 'pearson' data into 3 bins
    #     num_bins = 3
    #     binned = pd.cut(filtered_error_4pearson['pearson'], num_bins)
    #     categories = binned.cat.categories  # The bin intervals
    #     print(f'\n Pearson counts for {parameter} \n',
    #         filtered_error_4pearson.groupby(binned, observed=True)['pearson'].count())

    #     # For each bin, plot the violin for each BBH component
    #     # We use positions 0, 1, 2 (for the bins) with a small offset for each component.
    #     for i, cat in enumerate(categories):
    #         for idx, which_bbh in enumerate(range(3, 0, -1)):
    #             # Select data in the current bin for the given BBH component
    #             data = filtered_error_4pearson.loc[binned == cat, f'{parameter}_{which_bbh}'].dropna().values
    #             pos = i + offsets[idx]
    #             if len(data) > 0:
    #                 vp = ax.violinplot(data, positions=[pos], widths=0.15,
    #                                     showmeans=False, showmedians=False, showextrema=False)
    #                 for body in vp['bodies']:
    #                     body.set_facecolor(colors[idx])
    #                     body.set_edgecolor(colors[idx])
    #                     body.set_alpha(0.8)

    #     # Set x-axis ticks at the bin centers and label them using the midpoints of the bins
    #     ax.set_xticks(np.arange(num_bins))
    #     tick_labels = []
    #     for cat in categories:
    #         mid = (cat.left + cat.right) / 2
    #         tick_labels.append(f"{mid:.2f}")
    #     ax.set_xticklabels(tick_labels, rotation=45)

    #     # Set axis labels for each subplot
    #     ax.set_xlabel("Pearson Coefficient", fontsize=20)
    #     ax.set_ylabel(rf"$\delta {labels_param[lab]}\,/\,{labels_param[lab]}$", fontsize=20)
    #     lab += 1

    # # Create a global legend (placed above the subplots) using custom Line2D objects.
    # # Note: The order in the legend is reversed to map BBH 1 to the violin drawn for which_bbh=1.
    # legend_elements = [
    #     Line2D([0], [0], color=colors[2], lw=14, label='BBH 1'),
    #     Line2D([0], [0], color=colors[1], lw=14, label='BBH 2'),
    #     Line2D([0], [0], color=colors[0], lw=14, label='BBH 3')
    # ]
    # fig.legend(handles=legend_elements, loc='upper center', ncol=3, frameon=False, fontsize=28)

    # # Increase the y-limits to allow more vertical space and set uniform y-ticks
    # y_ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    # for ax in axs:
    #     ax.set_ylim(-1.5, 1.5)  # Increased y-limits; adjust as needed
    #     ax.set_yticks(y_ticks)

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(f"{SAVING_PATH}/pearson_coefficient/pearson_violin_combined_mode.png")
    # plt.show()


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
    error = pd.read_csv(f"{SAVING_PATH}/stats/errors_withpearson.csv")
    stat = pd.read_csv(f"{SAVING_PATH}/stats/stats_withpearson.csv")
    mode = pd.read_csv(f"{SAVING_PATH}/stats/mode.csv")
    median = pd.read_csv(f"{SAVING_PATH}/stats/median.csv")
    truth = pd.read_csv(f"{SAVING_PATH}/stats/truth.csv")
    pearson_pairs = pd.read_csv(f"{SAVING_PATH}/stats/pearson_coefficients_pairs.csv")

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
    pearson_statistic = 0
    for pair_id, total in pair_sums.items():
        pearson_statistic += total**2

    pearson_statistic = np.sqrt(pearson_statistic/3)
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



    # ##________________________TIME OVERLAP
    # # calculate total overlap
    # # somma in quadratura degli time_overlap 0,1,2
    # error_4timeoverlap = error
    # tot_overlap_sum = np.sqrt(np.sum([(stat[f'time_overlap_{i}'].values)**2 for i in range(num_overlapping_signals['BBH'])], axis=0))
    # error_4timeoverlap['tot_overlap'] = tot_overlap_sum
    # error_4timeoverlap[f'snr_{which_bbh-1}'] = stat[f'snr_{which_bbh-1}']
    # error_4timeoverlap['binned_values'] = pd.cut(error_4timeoverlap['tot_overlap'], 10)

    
    # tau_central_values = error.groupby('binned_values', observed=True)['tot_overlap'].mean()
    # error_mean = error_4timeoverlap.groupby('binned_values', observed=True)[f'M_BBH_{which_bbh}'].mean()
    # error_stds = error_4timeoverlap.groupby('binned_values', observed=True)[f'M_BBH_{which_bbh}'].std()
    
    
    # # Plotting with Seaborn
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(x=tau_central_values, y=error_mean, marker='o', label='Mean Values')

    # # Adding confidence interval band (±std deviation)
    # plt.fill_between(
    #     tau_central_values,
    #     np.array(error_mean) - np.array(error_stds),
    #     np.array(error_mean) + np.array(error_stds),
    #     color='b',
    #     alpha=0.2,
    #     label="±1 Std Dev"
    # )

    # # Labels and title
    # plt.xlabel("Time overlap")
    # plt.ylabel("Error")
    # plt.title(f"M_BBH_{which_bbh}")
    # plt.legend()
    # plt.savefig(f"{SAVING_PATH}/error_vs_overlap_plot.png")

    
    # ##________________________SNR
    # error_4snr = error
    # error_4snr['tot_overlap'] = tot_overlap_sum
    # error_4snr[f'snr_{which_bbh-1}'] = stat[f'snr_{which_bbh-1}']
    # error_4snr['binned_values'] = pd.cut(error_4snr[f'snr_{which_bbh-1}'], 10)


    # snr_central_values = error_4snr.groupby('binned_values', observed=True)[f'snr_{which_bbh-1}'].mean()    
    # error_mean = error_4snr.groupby('binned_values', observed=True)[f'M_BBH_{which_bbh}'].mean()
    # error_stds = error_4snr.groupby('binned_values', observed=True)[f'M_BBH_{which_bbh}'].std()

    # # calculate the 50 CL with np.quantile
    # error_50cl = error_4snr.groupby('binned_values', observed=True)[f'M_BBH_{which_bbh}'].apply(lambda x: np.quantile(x, 0.5)).head(10)
    
    
    # # Plotting with Seaborn
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(x=snr_central_values, y=error_mean, marker='o', label='Mean Values')

    # print len of 50cl and errror_mean
    #print(len(error_mean), len(error_50cl))

    # Adding confidence interval band (±std deviation)
    # plt.fill_between(
    #     snr_central_values,
    #     np.array(error_mean) - np.array(error_50cl),
    #     np.array(error_mean) + np.array(error_50cl),
    #     color='b',
    #     alpha=0.2,
    #     label="±1 Std Dev"
    # )

    # # Labels and title
    # plt.xlabel("SNR")
    # plt.ylabel("Error")
    # plt.title(f"M_BBH_{which_bbh}")
    # plt.legend()
    # plt.savefig(f"{SAVING_PATH}/error_vs_snr_bbh{which_bbh}_plot.png")


    ##_________________CORRELATION
    # error_4corr = error
    # error_4corr['tot_overlap'] = tot_overlap_sum
    # error_4corr[f'correlation'] = stat[f'correlation']
    # error_4corr['binned_values'] = pd.cut(error_4corr[f'correlation'], 10)
    # # tell me the number of elements in each bin
    # #print(error_4corr.groupby('binned_values', observed=True)[f'correlation'].count())

    # corr_central_values = error_4corr.groupby('binned_values', observed=True)[f'correlation'].mean()
    # error_mean = error_4corr.groupby('binned_values', observed=True)[f'M_BBH_{which_bbh}'].mean()
    # error_stds = error_4corr.groupby('binned_values', observed=True)[f'M_BBH_{which_bbh}'].std()

    # # calculate the 50 CL with np.quantile
    # error_50cl = error_4corr.groupby('binned_values', observed=True)[f'M_BBH_{which_bbh}'].apply(lambda x: np.quantile(x, 0.5)).head(10)

    # # Plotting with Seaborn
    # palette = sns.color_palette("coolwarm", as_cmap=False)
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(x=corr_central_values, y=error_mean, marker=markers.tortellini, label='Mean Values')

    # # Manually changing marker color using matplotlib
    # plt.plot(
    #     corr_central_values, 
    #     error_mean, 
    #     linestyle='None',  # No line
    #     marker=markers.tortellini,  # Keep the same marker
    #     markersize=10,
    #     markeredgecolor="blue",  # Change marker border color
    #     label="Custom Markers"
    # )

    # # Adding confidence interval band
    # plt.fill_between(
    #     corr_central_values,
    #     np.array(error_mean) - np.array(error_50cl),
    #     np.array(error_mean) + np.array(error_50cl),
    #     color='b',
    #     alpha=0.2,
    #     label="±1 Std Dev"
    # )


    # # Labels and title
    # plt.xlabel("Correlation")
    # plt.ylabel("Error")
    # plt.title(f"M_BBH_{which_bbh}")
    # plt.legend()
    # plt.savefig(f"{SAVING_PATH}/error_vs_corr_bbh{which_bbh}_plot.png")


    ##________________ plot all bbhs 1,2,3 correlations together with different colors

    # corr_central_values = error_4corr.groupby('binned_values', observed=True)[f'correlation'].mean()
    # error_mean_1 = error_4corr.groupby('binned_values', observed=True)[f'q_BBH_1'].mean()
    # error_stds_1 = error_4corr.groupby('binned_values', observed=True)[f'q_BBH_1'].std()
    # error_mean_2 = error_4corr.groupby('binned_values', observed=True)[f'q_BBH_2'].mean()
    # error_stds_2 = error_4corr.groupby('binned_values', observed=True)[f'q_BBH_2'].std()
    # error_mean_3 = error_4corr.groupby('binned_values', observed=True)[f'q_BBH_3'].mean()
    # error_stds_3 = error_4corr.groupby('binned_values', observed=True)[f'q_BBH_3'].std()

    # # calculate the 50 CL with np.quantile
    # error_50cl_1 = error_4corr.groupby('binned_values', observed=True)[f'q_BBH_1'].apply(lambda x: np.quantile(x, 0.5)).head(10)
    # error_50cl_2 = error_4corr.groupby('binned_values', observed=True)[f'q_BBH_2'].apply(lambda x: np.quantile(x, 0.5)).head(10)
    # error_50cl_3 = error_4corr.groupby('binned_values', observed=True)[f'q_BBH_3'].apply(lambda x: np.quantile(x, 0.5)).head(10)

    # # Plotting with Seaborn
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(x=corr_central_values, y=error_mean_1, marker='o', label='Mean Values BBH1', color='r')
    # sns.lineplot(x=corr_central_values, y=error_mean_2, marker='o', label='Mean Values BBH2', color='g')
    # sns.lineplot(x=corr_central_values, y=error_mean_3, marker='o', label='Mean Values BBH3', color='b')

    # # Adding confidence interval band
    # plt.fill_between(
    #     corr_central_values,
    #     np.array(error_mean_1) - np.array(error_50cl_1),
    #     np.array(error_mean_1) + np.array(error_50cl_1),
    #     color='r',
    #     alpha=0.2
    # )
    # plt.fill_between(
    #     corr_central_values,
    #     np.array(error_mean_2) - np.array(error_50cl_2),
    #     np.array(error_mean_2) + np.array(error_50cl_2),
    #     color='g',
    #     alpha=0.2
    # )
    # plt.fill_between(
    #     corr_central_values,
    #     np.array(error_mean_3) - np.array(error_50cl_3),
    #     np.array(error_mean_3) + np.array(error_50cl_3),
    #     color='b',
    #     alpha=0.2
    # )

    # # Labels and title
    # plt.xlabel("Correlation")
    # plt.ylabel("Error")
    # plt.title(f"M_BBH")
    # plt.legend()
    # plt.savefig(f"{SAVING_PATH}/error_vs_corr_bbh_all_plot_q.png")



    #_________________________PEARSON
    # Calculate Pearson coefficient related variables
    for parameter in parameters:
        error_4pearson = error_mode

        error_4pearson[f'pearson'] = pearson_statistic

        mask = (error_4pearson[[f'{parameter}_1', f'{parameter}_2', f'{parameter}_3']] < 1).all(axis=1) & (error_4pearson[f'pearson']<0.5)


        # Apply the mask to get filtered data
        filtered_error_4pearson = error_4pearson[mask]

        # Violin plot with filtered errors
        num_bins = 3
        #binned = pd.qcut(filtered_error_4pearson[f'pearson'], q=num_bins, duplicates='drop')

        binned = pd.cut(filtered_error_4pearson[f'pearson'], num_bins)
        # print number of samples in each bin
        print('\n pearson\n', filtered_error_4pearson.groupby(binned, observed=True)[f'pearson'].count())

        # # Randomly sample 600 data points from each bin
        # sampled_data = (
        #     filtered_error_4pearson.groupby(binned, observed=True)
        #     .apply(lambda x: x.sample(n=min(600, len(x)), random_state=42))  # Ensures we don't exceed available samples
        #     .reset_index(drop=True)
        # )

        # print('\n pearson\n', sampled_data.groupby(binned, observed=True)[f'pearson'].count())


        fig, ax = plt.subplots(figsize=(12, 8))  # Improved figure dimensions
        #colors = ['#E88873', '#A8C69F', '#9B93B4'] purple, green and orange, nice but not extraordinary
        colors = ['#AB87FF', '#FCD581', '#43C7BE']  # purple, yellow and bluegreen I LOVE IT
        #colors = ['#FFA987', '#E54B4B', '#6A0136'] #sul rosa rosso bello, statement femminista
        #colors = ['#F2B880', '#C98686', '#966B9D'] #classy purple, pink and beige
        colors = ['#ffa62b', '#82c0cc', '#1E6091'] # PIU' QUOTATA
        #colors = ['#07beb8', '#68d8d6', '#9ceaef'] 
        #colors = ['#D5B942', '#22AED1', '#016FB9']
       # colors = ['#A1176E', '#7A0033', '#2E0219']
        #colors = ['#D9ED92', '#52B69A', '#1A759F']
       #colors_edge = ['#99D98C', '#34A0A4', '#184E77']
        colors = ['#ffa62b', '#49A3C9', '#9E5793']

        for idx, which_bbh in enumerate(range(3, 0, -1)):

            filtered_errors = filtered_error_4pearson[f'{parameter}_{which_bbh}']

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

        '''
        error_4pearson['binned_values'] = pd.cut(error_4pearson[f'pearson'], 10)
        # tell me the number of elements in each bin
        print('\n pearson\n', error_4pearson.groupby('binned_values', observed=True)[f'pearson'].count())

        pearson_central_values = error_4pearson.groupby('binned_values', observed=True)[f'pearson'].mean()
        error_mean_1 = error_4pearson.groupby('binned_values', observed=True)[f'M_BBH_1'].mean()
        error_stds_1 = error_4pearson.groupby('binned_values', observed=True)[f'M_BBH_1'].std()
        error_mean_2 = error_4pearson.groupby('binned_values', observed=True)[f'M_BBH_2'].mean()
        error_stds_2 = error_4pearson.groupby('binned_values', observed=True)[f'M_BBH_2'].std()
        error_mean_3 = error_4pearson.groupby('binned_values', observed=True)[f'M_BBH_3'].mean()
        error_stds_3 = error_4pearson.groupby('binned_values', observed=True)[f'M_BBH_3'].std()

        # calculate the 50 CL with np.quantile
        error_50cl_1 = error_4pearson.groupby('binned_values', observed=True)[f'M_BBH_1'].apply(lambda x: np.quantile(x, 0.5)).head(10)
        error_50cl_2 = error_4pearson.groupby('binned_values', observed=True)[f'M_BBH_2'].apply(lambda x: np.quantile(x, 0.5)).head(10)
        error_50cl_3 = error_4pearson.groupby('binned_values', observed=True)[f'M_BBH_3'].apply(lambda x: np.quantile(x, 0.5)).head(10)

        # Plotting with Seaborn
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=pearson_central_values, y=error_mean_1, marker='o', label='Mean Values BBH1', color='r')
        sns.lineplot(x=pearson_central_values, y=error_mean_2, marker='o', label='Mean Values BBH2', color='g')
        sns.lineplot(x=pearson_central_values, y=error_mean_3, marker='o', label='Mean Values BBH3', color='b')

        # Adding confidence interval band
        plt.fill_between(
            pearson_central_values,
            np.array(error_mean_1) - np.array(error_50cl_1),
            np.array(error_mean_1) + np.array(error_50cl_1),
            color='r',
            alpha=0.2
        )
        plt.fill_between(
            pearson_central_values,
            np.array(error_mean_2) - np.array(error_50cl_2),
            np.array(error_mean_2) + np.array(error_50cl_2),
            color='g',
            alpha=0.2
        )
        plt.fill_between(
            pearson_central_values,
            np.array(error_mean_3) - np.array(error_50cl_3),
            np.array(error_mean_3) + np.array(error_50cl_3),
            color='b',
            alpha=0.2
        )

        # Labels and title
        plt.xlabel("Pearson Coefficient")
        plt.ylabel("Error")
        plt.title(f"M_BBH")
        plt.legend()
        plt.savefig(f"training_results/{MODEL_NAME}/error_vs_pearson_bbh_all_plot.png")

        '''



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


    # joyplot
    import joypy
    pops = ['snr10-30', 'snr30-60', 'snr60-90', 'snr90-120', 'snr120-150']
    par = 'q'
    data = {}

    for pop in pops:
        data[pop] = pd.read_csv(f"{MAIN_PATH}/test-pop_{pop}/stats/errors_withpearson.csv")
    

    # make a padas with the "M_BBH_1" errors from all the populations
    M_snr_1 = pd.DataFrame()
    M_snr_2 = pd.DataFrame()
    M_snr_3 = pd.DataFrame()

    for pop in pops:
        M_snr_1[pop] = data[pop][f'{par}_BBH_1']
        M_snr_2[pop] = data[pop][f'{par}_BBH_2']
        M_snr_3[pop] = data[pop][f'{par}_BBH_3']



    fig, ax = plt.subplots(figsize=(10, 6))
    #joypy.joyplot(M_snr_1, ax=ax, color=colors[2], alpha=0.6, x_range=[-1, 1])
    #joypy.joyplot(M_snr_2, ax=ax, color=colors[1], alpha=0.6, x_range=[-1, 1])
    joypy.joyplot(M_snr_3, ax=ax, color=colors[0], alpha=0.6, x_range=[-1, 1])
    ax.set_xlabel("Error on M_BBH_1")
    ax.set_ylabel("Density")
    ax.set_title("Error on M_BBH_1 Joyplot")

    plt.savefig(f"{MAIN_PATH}/joyplot.png")
    

    
"""Implementation of a PP Plot"""
import os
import scipy
import shutil
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt


from scipy.stats import kstest, beta
from matplotlib.colors import LinearSegmentedColormap


def latexify(plot_func):
    """A decorator to apply LaTeX styling to matplotlib plots.
    """
    def wrapper_plot(*args, **kwargs):
        plt.rcParams["font.size"] = 16

        #use latex if available on the machine
        if shutil.which("latex"): 
            plt.rcParams.update({"text.usetex": True, 
                                 "font.family": "modern roman", 
                                 'text.latex.preamble': r'\usepackage{amsmath}'})
            
        return plot_func(*args, **kwargs)
    return wrapper_plot



converter = {'M'                  : 'M',
             'q'                  : 'q',
             'Mchirp'             : '\mathcal{M}',
             'tcoal'              : 't'}


def get_latex_label(name):
    par_name, kind, num = name.split('_')
    return rf'${converter[par_name]}_{{{kind},{num}}}$'


kenn_color     = "#DE429D"
hyperion_color = "#8DA0E9"

@latexify
def make_pp_plot(posteriors=None, true_parameters=None, saved_data=None, num_quantiles=1001, out_dir=None, colormap='coolwarm_r'):
    """
    Create a PP (Probability-Probability) plot for evaluating posterior distributions.
    A separate PP line is drawn for each parameter, with KS test results in the legend.

    Parameters:
    - posteriors (list of dict): A list of dictionaries where each dictionary contains 
      posterior samples for a given event.
    - true_parameters (list of dict): A list of dictionaries where each dictionary contains 
      the true parameter values for a given event.
    - num_quantiles (int): Number of quantiles to compute for the cumulative probabilities.

    Returns:
    - None: Displays the PP plot.
    """
    
    out_dir = os.getcwd() if out_dir is None else out_dir


    if saved_data is not None:
        pp_data = np.load(saved_data, allow_pickle=True)
        quantiles, ecdf, num_posteriors = pp_data["quantiles"], pp_data["param_ecdfs"], pp_data["num_posteriors"]
        param_ecdfs = ecdf.item()  
        param_names = list(param_ecdfs.keys())
        physical_p  = np.unique([p.split('_')[0] for p in param_names])

    else:
        assert len(posteriors) == len(true_parameters), "Mismatch in number of events between posteriors and true parameters."
        num_posteriors = len(posteriors)

        # Get all parameter names
        param_names = true_parameters[0].keys()
        physical_p = np.unique([p.split('_')[0] for p in param_names])


        quantiles = np.linspace(0, 1, num_quantiles + 1)
        param_ecdfs = {param: [] for param in param_names}

        for i in range(1, len(posteriors)):
            posterior = posteriors[i]
            true_param = true_parameters[i]
            for param_name in param_names:
                
                true_value = true_param[param_name]
                samples = np.array(posterior[param_name])

                # Calculate the cumulative fraction of samples <= true_value
                fraction = np.mean(samples <= true_value)
                param_ecdfs[param_name].append(fraction)

    
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [kenn_color, hyperion_color], len(physical_p))

    # Plot each parameter's PP line
    plt.figure(figsize=(8, 8))  # Square figure

    # Plot the perfect calibration line
    plt.plot(quantiles, quantiles, linestyle="--", color="k")#, label="Perfect Calibration")

    
    # Add 1, 2, 3-sigma contours based on ECDF
    CL = [0.997, 0.95, 0.68]
    sigma = [1, 2, 3]
    alpha = [0.05, 0.1, 0.15]
    for i, cl in enumerate(CL):
        n = num_posteriors
        k = np.arange(0, n + 1)
        p = k / n
        lower, upper = beta.interval(cl, k + 1, n - k + 1)
        plt.fill_between(p, np.clip(lower, 0, 1), np.clip(upper, 0, 1), alpha=alpha[i], color= 'k')#, label=f"{sigma[i]} σ region" )
    

    #----- PP plot --------
    # get colors and linestyles
    linestyles = {1: 'solid', 2: 'dashed', 3: 'dashdot'}
    #colors = {p: cmap(i) for i, p in enumerate(physical_p)}
    cmap = sns.color_palette("magma", n_colors=len(physical_p)+2)
    colors = {p: cmap[i+1] for i, p in enumerate(physical_p)}
    
    p_values = []
    for p in physical_p:
        for num in range(1, len(param_names)//len(physical_p)+1):
            param_name = f"{p}_BBH_{num}"
    
            cumulative_fractions = np.array(param_ecdfs[param_name])
            ecdf = [(cumulative_fractions <= q).mean() for q in quantiles]

            # Perform KS test
            _ , p_value = kstest(cumulative_fractions, 'uniform')
            p_values.append(p_value)

            # Plot the ECDF
            plt.plot(quantiles, ecdf, color=colors[p], linestyle=linestyles[int(num)], label=f"{get_latex_label(param_name)} ({p_value:.3f})")

    combined_p_value = scipy.stats.combine_pvalues(p_values, method='fisher')[1]

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("$p$")
    plt.ylabel("$CDF(p)$")
    plt.title(f"PP Plot with {num_posteriors} injections (combined $p$$-$$value: {combined_p_value:.3f}$)", fontdict={'fontsize': 16})
    legend_fontsize = 11
    plt.legend(title="Parameters and KS Test", fontsize=legend_fontsize, title_fontsize=legend_fontsize+1)
    plt.grid(True, linestyle=':', alpha=0.8)

    # Set the aspect ratio to 1:1
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'{out_dir}/PP_plot.pdf', dpi=200, bbox_inches='tight')
    plt.show()

    #save data
    if saved_data is None:
        np.savez(f'{out_dir}/pp_data.npz', quantiles=quantiles, param_ecdfs=param_ecdfs, num_posteriors=len(posteriors))

    
    




if __name__=='__main__':

    par = "M_BBH_1"
    print(get_latex_label(par))


    plt.figure()
    plt.xlabel(get_latex_label(par))
    plt.savefig('test.png')

    # Example posterior samples for 2 events
    posteriors = [
        {"Mchirp_BBH_1": np.random.normal(35, 5, 1000), "q_BBH_1": np.random.uniform(0.5, 1.0, 1000)},
        {"Mchirp_BBH_1": np.random.normal(25, 3, 1000), "q_BBH_1": np.random.uniform(0.6, 0.9, 1000)}
    ]

    # True parameters for the same events
    true_parameters = [
        {"Mchirp_BBH_1": 34.5, "q_BBH_1": 0.8},
        {"Mchirp_BBH_1": 26.0, "q_BBH_1": 0.7}
    ]

    # Generate the PP plot
    make_pp_plot(posteriors, true_parameters)





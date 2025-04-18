
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-m", "--m_min", default=100, help="Minimum mass of the primary black hole. (Default: 100 Msun)")
    parser.add_option("-s", "--snr_max", default=300, help="Maximum network SNR. (Default: 300)")
    (options, args) = parser.parse_args()
   
    m_min   = int(options.m_min)
    snr_max = int(options.snr_max)

    # Leggi i dati
    df = pd.read_csv(f'snr_results_m_min_{m_min}.csv')
    df['distance'] /= 1e3  # Rescaling to Gpc
    
    #maschera i dati
    df = df[df['network_snr'] <= snr_max]  # SNR max


    # Rinomina la colonna 'distance' in 'distance [Gpc]'
    df.rename(columns={'distance'   : '$d_L$ $[Gpc]$',
                       'mass1'      : '$m_1^z$ $[M_{\odot}]$',
                       'mass2'      : '$m_2^z$ $[M_{\odot}]$',
                       'network_snr': '$\\rho_{net}$',
                       }, inplace=True)
    
    # Elimina la colonna 'redshift'
    df = df.drop(columns=['z'])
    
    # Sottocampiona i dati se sono troppi
    if len(df) > 10_000:  # Ad esempio, se hai più di 10.000 punti
        df = df.sample(frac=0.5, random_state=42)  # Prendi solo il 10% dei dati

    
    # Funzione per calcolare mediana e intervallo di confidenza
    def median_and_cl(data):
        median = np.median(data)
        lower = np.percentile(data, 16)  # 16th percentile
        upper = np.percentile(data, 84)  # 84th percentile
        return median, lower, upper

    # Crea il pairplot con istogrammi
    plot = sns.pairplot(df, kind='hist', diag_kind='kde', plot_kws={'alpha': 0.5, 'bins':50}, corner=True)
    print('plot done')
    # Aggiungi i contorni KDE sui grafici off-diagonal
    # Crea la mappa di colori dalla palette 'crest'
    cmap = sns.color_palette('twilight', as_cmap=True)
    for i, j in zip(*np.tril_indices_from(plot.axes, -1)):
        sns.kdeplot(data=df, x=df.columns[j], y=df.columns[i], ax=plot.axes[i, j], levels=5, cmap=cmap)

    # Attiva i minorticks e annota la mediana e il 68% CL
    for i, ax in enumerate(plot.diag_axes):  # Solo sugli assi diagonali
        column = df.columns[i]
        print(column)
        data = df[column].dropna()
        
        # Calcola la mediana e il 68% CL
        median, lower, upper = median_and_cl(data)
        
        # Annotazioni nei grafici
        text = rf'${median:.2f}^{{+{upper:.2f}}}_{{-{lower:.2f}}}$'
        ax.annotate(text, xy=(0.5, 0.9), xycoords='axes fraction', ha='center')

        ax.minorticks_on()  # Attiva i minorticks
        ax.tick_params(which='minor', length=4, color='r')

    plt.savefig(f'snr_pairplot_m_min_{m_min}.png', dpi=300, bbox_inches='tight')
    plt.show()
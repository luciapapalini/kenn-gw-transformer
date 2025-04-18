import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 14
plt.rcParams.update({"text.usetex": True,
                        "font.family": "modern roman"})

if __name__=="__main__":

    # read the data
    f_ligo, asd_ligo = np.loadtxt("ligo.txt", unpack=True)
    f_virgo, asd_virgo = np.loadtxt("virgo.txt", unpack=True)
    f_et, _, _, asd_et= np.loadtxt("et.txt", unpack=True)


    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the curves
    ax.loglog(f_ligo, asd_ligo, color='black', label=r'\textsc{LIGO A+}')
    ax.loglog(f_virgo, asd_virgo, color='purple', label=r'\textsc{Virgo O5}')
    ax.loglog(f_et, asd_et, color='green', label=r'\textsc{ET-D}')

    # Labels
    ax.set_xlabel(r"$\mathit{f}~[\mathrm{Hz}]$", fontsize=14)
    ax.set_ylabel(r"$\mathrm{ASD}~[1/\sqrt{\mathrm{Hz}}]$", fontsize=14)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make x and y axis spines thicker
    ax.spines['bottom'].set_linewidth(1.1)
    ax.spines['left'].set_linewidth(1.1)

    # Grid (keep it light and dashed)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.7)

    # Legend with a frame
    #add legend on top in one line no alpha
    ax.legend(loc="upper center", frameon=True, fontsize=13, ncols=3)

    #ax.legend(loc="upper right", frameon=True, fontsize=12)

    # Show plot
    plt.savefig("asd_plot.pdf", dpi=200, bbox_inches='tight')
    plt.show()
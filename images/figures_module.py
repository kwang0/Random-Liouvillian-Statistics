import matplotlib.pyplot as plt
import matplotlib
import cycler
import numpy as np
                                                                                                                                                        
cmap = matplotlib.cm.get_cmap('viridis_r')
cmap2 = matplotlib.cm.get_cmap('inferno')                                                                                                                                                                          
colors = [cmap(n) for n in np.linspace(0, 1, 9)]            
colors2 = [cmap2(n) for n in np.linspace(0, 1, 9)]            

def prepare_standard_figure(nrows=1, ncols=1, sharex=False, sharey=False, width=3.375, aspect_ratio=1.61):

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath},')
    plt.rc('font', family='serif', size=6)
    plt.rc('lines', linewidth=1)
    plt.rc('xtick', labelsize='medium', direction="in", top=True)
    plt.rc('ytick', labelsize='medium', direction="in", right=True)
    plt.rc('legend', fontsize='small', numpoints=1, frameon=False, handlelength=1)
    plt.rc('axes', linewidth=0.5)
    plt.rc('errorbar', capsize=2)
    plt.rc('savefig', dpi=800)

    # change color scheme
#    colors__ = ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#990000"]
    colors__ = ["#ffc30f", "#ff5733", "#c70039", "#900c3f", "#581845",'k']

    plt.rcParams['axes.prop_cycle'] = cycler.cycler(color=colors)

    fig_size = (width, width/aspect_ratio)
    f1, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, figsize=fig_size)
    return f1, axs


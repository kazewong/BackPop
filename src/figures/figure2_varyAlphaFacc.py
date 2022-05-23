import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

models = ['A22_02', 'A22', 'A22_5', 'KW']

def get_data(model):
    if model == 'KW':
        bpp = pd.read_hdf('{}_rerun.h5'.format(model), key='bpp')
        initC = pd.read_hdf('{}_rerun.h5'.format(model), key='initC')
    else:    
        bpp = pd.read_hdf('{}_rerun.hdf'.format(model), key='bpp')
        initC = pd.read_hdf('{}_rerun.hdf'.format(model), key='initC')
    
    mergers = bpp.loc[(bpp.kstar_1 == 14) & (bpp.kstar_2 == 14) & (bpp.evol_type == 3)]
    formation = bpp.loc[(bpp.bin_num.isin(mergers.bin_num)) & (bpp.kstar_1 == 14) & (bpp.kstar_2 == 14)].groupby('bin_num').first().reset_index()   
    bpp = []
    return initC.loc[initC.bin_num.isin(mergers.bin_num)], mergers, formation

initC_list = []
merger_list = []
formation_list =[]
for m in models:
    dat = get_data(model=m)
    initC_list.append(dat[0])
    merger_list.append(dat[1])
    formation_list.append(dat[2])

labels = [r'$\alpha=0.2$, $f_{\rm{acc}}=0.5$', 
          r'$\alpha=1$, $f_{\rm{acc}}=0.5$', 
          r'$\alpha=5$, $f_{\rm{acc}}=0.5$', 
          r'$\alpha=\rm{var}$, $f_{\rm{acc}}=\rm{var}$']

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 3))

for ax, initC, label, ii in zip(axes, initC_list, labels, range(len(labels))):
    p, x_edges, y_edges, im = ax.hist2d(x=initC.mass_1.values, y=initC.mass_2.values, 
                                        bins=[np.linspace(50, 150, 50), np.linspace(35, 150, 50)], 
                                        label=label, density=True, cmap='mako_r', 
                                        norm=colors.LogNorm(vmin=1e-6, vmax=6e-2))
    
    if ii > 2:
        ax.spines['bottom'].set(linewidth=5)
        ax.spines['top'].set(linewidth=5)
        ax.spines['right'].set(linewidth=5)
        ax.spines['left'].set(linewidth=5)
    ax.set_xlabel(r'$M_{1,\rm{ZAMS}}$ [M$_{\odot}$]', size=14)
    ax.set_xlim(50,150)
    ax.set_ylim(35,150)
    ax.set_title(label, size=14)
    if ii == 0:
        ax.set_ylabel(r'$M_{2,\rm{ZAMS}}$ [M$_{\odot}$]', size=14)
        
cb = fig.colorbar(im, ax=axes, fraction=0.05, pad=0.01, aspect=15)
cb.set_label(label=r'dN/$dM_{1,\rm{ZAMS}}dM_{2,\rm{ZAMS}}$', size=12)
plt.savefig('compare_fixed_variable.png', facecolor='white', dpi=180, bbox_inches = 'tight')
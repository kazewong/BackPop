import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy

import matplotlib as mpl
import corner
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.lines import Line2D

params = {'axes.labelsize': 32,
          'font.family': 'serif',
          'font.serif': 'Computer Modern Raman',
          'font.size': 32,
          'axes.linewidth': 2,
          'legend.fontsize': 20,
          'xtick.labelsize': 20,
          'xtick.top': True,
          'xtick.direction': "in",
          'ytick.labelsize': 20,
          'ytick.right': True,
          'ytick.direction': "in",
          'axes.grid' : False,
          'text.usetex': True,
          'savefig.dpi' : 100,
          'lines.markersize' : 14,
#          'axes.formatter.useoffset': False,
          'axes.formatter.limits' : (-3,3)}

#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

mpl.rcParams.update(params)

def fill_corner(fig,data,ndim,axis,color='C0',true_val=None,zorder=1,lw=3,style='-',smooth1d = 0.01, smooth2d=0.01,no_fill_contours=False,fill_contours=True,alpha=1,levels=[0.68,0.95]):
    for i in range(ndim):
        for j in range(i+1):
            ax = fig.axes[np.sum(np.arange(i+1))+j]
            if i==j:
                ax.hist(data[i],bins=axis[i],histtype='step',lw=lw,color=color,density=True,zorder=zorder)
                ylim = ax.get_ylim()
                ax.set_xlim(axis[i][0],axis[i][-1])
                ax.set_ylim(ylim[0],ylim[1])
                if true_val is not None:
                    ax.vlines(true_val[i],ylim[0],ylim[1],color=color,lw=lw,zorder=zorder)
            elif j<i:
                corner.hist2d(data[j],data[i],bins=[axis[j],axis[i]],smooth=smooth2d,plot_datapoints=False,plot_density=False,ax=ax,levels=levels,fill_contours=fill_contours,smooth1d = smooth1d,color=color,no_fill_contours=True,contour_kwargs={'linewidths':lw,'zorder':zorder,'linestyles':style,'colors':color})
                ax.set_ylim(axis[i][0],axis[i][-1])
                ax.set_xlim(axis[j][0],axis[j][-1])
                if true_val is not None:
                    ax.scatter(true_val[j],true_val[i],color=color,marker='*',s=100,zorder=20)
                    ax.vlines(true_val[j],ax.get_ylim()[0],ax.get_ylim()[1],color=color,lw=3)
                    ax.hlines(true_val[i],ax.get_xlim()[0],ax.get_xlim()[1],color=color,lw=3)


def get_posterior(value, coord,cut = 1e-2):
    a = copy.deepcopy(coord)
    a[...,0] = 10**a[...,0]
    a[...,1] = 10**a[...,1]

    a[...,[1,0]] = np.sort(a[...,[0,1]],axis=2)
    a[...,2] = a[...,2]
    a[...,4] = a[...,4]
    a[...,7] = a[...,7]
    b = value

    criteria = cut

    count = np.sum((b<criteria).astype(int),axis=1)
    weight = np.ones(b.shape)/count[:,None]
    weight[np.invert(np.isfinite(weight))] = 0

    index = np.where((b<criteria))
    return a[index],b[index],weight[index],index

i = 0
data = h5py.File('/mnt/home/wwong/ceph/GWProject/ProgenitorCatalogue/GWTC3_bse_event_1_merged.hdf5')
GWTC = np.load('/mnt/ceph/users/wwong/GWProject/GWTC/Processed/Combined_GWTC_m1m2chieffz.npz')
processed_data = h5py.File('/mnt/home/wwong/ceph/GWProject/ProgenitorCatalogue/GWTC3_bse_event_1_merged_process.hdf5')
m1_obs = np.median(GWTC['m1'],axis=1)
m2_obs = np.median(GWTC['m2'],axis=1)
z_obs = np.median(GWTC['z'],axis=1)
q = m2_obs/m1_obs
m1_pred = processed_data['m1']
m2_pred = processed_data['m2']
t_merge = processed_data['t_merge']
observables = np.stack([GWTC['m1'],GWTC['m2'],GWTC['z']],axis=2)
posterior_array = []
weight_array = []
observable_array = []
quantile_array = []

a,b,w,index = get_posterior(data['result_value'][i],data['result_coord'][i],1e-1)
posterior_array.append(a)
weight_array.append(w)
observable_array.append(observables[i][index[0]])
quantile_array.append(np.quantile(a,[0.5-0.95/2,0.5,0.5+0.95/2],axis=0))
quantile_array = np.array(quantile_array)

index = 0
Ndim = 10
xlabel = [r'$M_1$',r'$M_2$',r'$\log{t_{\rm orb}}\ {\rm [days]}$',r'$e$',r'$\log{Z}$',r'$\alpha$',r'$f_{\rm acc}$',r'$q_{\rm crit, 3}$',r'$q_{\rm crit, 4}$','$\sigma$']
lower_bound = [10.0 , 10.0 , np.log10(50) , 0.0, np.log10(0.0002), 0.25, 0.0, 1.0, 1.0, np.log10(40)]
upper_bound = [120.0, 120.0, np.log10(5000), 1.0, np.log10(0.02)  , 5.0 , 1.0, 4.0, 4.0, np.log10(265)]
axis = np.array([np.linspace(lower_bound[i],upper_bound[i],20) for i in range(Ndim)])

fig = plt.figure(figsize=(40,40))
grid = plt.GridSpec(Ndim,Ndim,wspace=0.1,hspace=0.1)

for i in range(Ndim):
    for j in range(i+1):      
        ax = fig.add_subplot(grid[i,j])

fill_corner(fig,posterior_array[0].T,Ndim,axis,style=['-.','--','-'],fill_contours=False,lw=4,levels=[0.68,0.95,0.995],smooth2d=0.1)

counter = 0
for i in range(Ndim):
    for j in range(i+1):      
        ax = fig.axes[counter]
        if i!=Ndim-1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(xlabel[j],fontsize=30)
            ax.set_xticklabels(np.around(ax.get_xticks(),1),rotation=45,fontsize=30)
        if j!=0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(xlabel[i],fontsize=30)
            ax.set_yticklabels(np.around(ax.get_yticks(),1),rotation=45,fontsize=30)
        counter += 1
        ax.tick_params(axis="both",direction="in",which='major',width=2.5,length=6,right=True,top=True)
        ax.tick_params(axis="both",direction="in",which='minor',width=1.5,length=3,right=True,top=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
fig.savefig('./GW150914_corner.pdf',bbox_inches='tight')
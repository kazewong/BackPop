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

def fill_corner(fig,data,ndim,axis,color='C0',true_val=None,zorder=1,lw=3,style='-',smooth1d = 0.01,no_fill_contours=False,fill_contours=True,alpha=1,levels=[0.68,0.95]):
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
                corner.hist2d(data[j],data[i],bins=[axis[j],axis[i]],smooth=1,plot_datapoints=False,plot_density=False,ax=ax,levels=levels,fill_contours=fill_contours,smooth1d = smooth1d,color=color,no_fill_contours=True,contour_kwargs={'linewidths':lw,'zorder':zorder,'linestyles':style,'colors':color})
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

a,b,w,index = get_posterior(data['result_value'][0],data['result_coord'][0],1e-1)
posterior_array.append(a)
weight_array.append(w)
observable_array.append(observables[i][index[0]])
quantile_array.append(np.quantile(a,[0.5-0.95/2,0.5,0.5+0.95/2],axis=0))
quantile_array = np.array(quantile_array)
plt.figure(figsize=(10,9))
percentile = [0.7,0.9]
corner.hist2d(m1_pred[()],m2_pred[()],weights=weight_array[0],color='C0',plot_density=False,labels='Reprojected',levels=percentile)
corner.hist2d(observable_array[0][:,0],observable_array[0][:,1],weights=weight_array[0],color='C1',plot_density=False,labels='Correponding',levels=percentile)
corner.hist2d(observables[i][:,0],observables[i][:,1],color='C2',plot_density=False,labels='Data',levels=percentile)
plt.xlabel(r'$M_1$',fontsize=30)
plt.ylabel(r'$M_2$',fontsize=30)
plt.title('GW150914')
custom_lines = [Line2D([0], [0], color='C0', lw=4),
                Line2D([0], [0], color='C1', lw=4),
                Line2D([0], [0], color='C2', lw=4)]
plt.legend(custom_lines, ['Reprojected','Corresponding','Data'],fontsize=20)
plt.savefig('GW150914_reprojection.pdf')
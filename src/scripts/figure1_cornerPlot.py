import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit
import copy
from astropy.cosmology import Planck18

import matplotlib as mpl
import corner
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, cdist
import paths

params = {
    "axes.labelsize": 32,
    "font.family": "serif",
    "font.size": 32,
    "axes.linewidth": 2,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "xtick.top": True,
    "xtick.direction": "in",
    "ytick.labelsize": 20,
    "ytick.right": True,
    "ytick.direction": "in",
    "axes.grid": False,
    "text.usetex": False,
    "savefig.dpi": 100,
    "lines.markersize": 14,
    "axes.formatter.limits": (-3, 3),
    "mathtext.fontset": "dejavuserif",
}


mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]  # for \text command

mpl.rcParams.update(params)


def fill_corner(
    fig,
    data,
    ndim,
    axis,
    color="C0",
    true_val=None,
    zorder=1,
    lw=3,
    style="-",
    smooth1d=0.01,
    smooth2d=0.01,
    no_fill_contours=False,
    fill_contours=True,
    alpha=1,
    levels=[0.68, 0.95],
):
    for i in range(ndim):
        for j in range(i + 1):
            ax = fig.axes[np.sum(np.arange(i + 1)) + j]
            if i == j:
                ax.hist(
                    data[i],
                    bins=axis[i],
                    histtype="step",
                    lw=lw,
                    color=color,
                    density=True,
                    zorder=zorder,
                )
                ylim = ax.get_ylim()
                ax.set_xlim(axis[i][0], axis[i][-1])
                ax.set_ylim(ylim[0], ylim[1])
                if true_val is not None:
                    ax.vlines(
                        true_val[i], ylim[0], ylim[1], color=color, lw=lw, zorder=zorder
                    )
            elif j < i:
                corner.hist2d(
                    data[j],
                    data[i],
                    bins=[axis[j], axis[i]],
                    smooth=smooth2d,
                    plot_datapoints=False,
                    plot_density=False,
                    ax=ax,
                    levels=levels,
                    fill_contours=fill_contours,
                    smooth1d=smooth1d,
                    color=color,
                    no_fill_contours=True,
                    contour_kwargs={
                        "linewidths": lw,
                        "zorder": zorder,
                        "linestyles": style,
                        "colors": color,
                    },
                )
                ax.set_ylim(axis[i][0], axis[i][-1])
                ax.set_xlim(axis[j][0], axis[j][-1])
                if true_val is not None:
                    ax.scatter(
                        true_val[j],
                        true_val[i],
                        color=color,
                        marker="*",
                        s=100,
                        zorder=20,
                    )
                    ax.vlines(
                        true_val[j],
                        ax.get_ylim()[0],
                        ax.get_ylim()[1],
                        color=color,
                        lw=3,
                    )
                    ax.hlines(
                        true_val[i],
                        ax.get_xlim()[0],
                        ax.get_xlim()[1],
                        color=color,
                        lw=3,
                    )


def get_posterior(value, coord, cut=1e-2):
    a = copy.deepcopy(coord)
    a[..., 0] = 10 ** a[..., 0]
    a[..., 1] = 10 ** a[..., 1]

    a[..., [1, 0]] = np.sort(a[..., [0, 1]], axis=2)
    a[..., 2] = a[..., 2]
    a[..., 4] = a[..., 4]
    a[..., 7] = a[..., 7]
    b = value

    criteria = cut

    count = np.sum((b < criteria).astype(int), axis=1)
    weight = np.ones(b.shape) / count[:, None]
    weight[np.invert(np.isfinite(weight))] = 0

    index = np.where((b < criteria))
    return a[index], b[index], weight[index], index


GWTC = np.load(paths.data / "GWTC3_LVK_posterior.npz")
m1_obs = np.median(GWTC["m1"], axis=1)
m2_obs = np.median(GWTC["m2"], axis=1)
z_obs = np.median(GWTC["z"], axis=1)
q = m2_obs / m1_obs
observables = np.stack([GWTC["m1"], GWTC["m2"], GWTC["z"]], axis=2)
z_axis = np.linspace(0.01, 1000, 10000)
lookback_time = Planck18.lookback_time(z_axis).value
z_func = interp1d(lookback_time, z_axis, bounds_error=False, fill_value=(0, 1000))
lookback_time_func = interp1d(
    z_axis, lookback_time, bounds_error=False, fill_value=(0, lookback_time.max())
)


def sample_z(kde, m1m2):
    kde_mass = kde.resample(int(m1m2.shape[0] * 10))
    distance = cdist(kde_mass[:2].T, m1m2)
    return kde_mass[2, np.argmin(distance, axis=0)]


i = 0
processed_data = h5py.File(paths.data / "GWTC3_mcmc_processed.hdf5", "r")

m1_pred = processed_data["m1"][i]
m2_pred = processed_data["m2"][i]
t_merge = processed_data["t_merge"][i][()] / 1000

posterior_array = []
observable_array = []
quantile_array = []

posterior_array.append(processed_data["coord_array"][i][()].T)
m1m2_array = np.stack([m1_pred[()], m2_pred[()]]).T
obs_kde = gaussian_kde(observables[0].T)

observed_z = []
for i in range(int(np.ceil(m1m2_array.shape[0] / 1000))):
    observed_z.append(sample_z(obs_kde, m1m2_array[i * 1000 : (i + 1) * 1000]))
observed_z = np.concatenate(observed_z)
z_pred = z_func(t_merge + lookback_time_func(observed_z))
posterior_array[0] = np.concatenate([posterior_array[0], z_pred[:, None]], axis=1)
m2, m1 = np.sort(posterior_array[0][:, :2], axis=1).T
posterior_array[0][:, 0] = 10 ** m1
posterior_array[0][:, 1] = 10 ** m2
posterior_array = posterior_array[0].T

index = i
Ndim = 7
xlabel = [
    r"$m_{1,{\rm ZAMS}}$",
    r"$m_{2,{\rm ZAMS}}$",
    r"$\log{Z}$",
    "$z$",
    r"$\alpha$",
    r"$f_{\rm acc}$",
    r"$q_{\rm crit, 3}$",
]
lower_bound = [10.0, 10.0, np.log10(0.0001), 0, 0.1, 0.0, 1.0, 1.0]
upper_bound = [150.0, 150.0, np.log10(0.02), 4, 10.0, 1.0, 10.0, 10.0]
axis = np.array([np.linspace(lower_bound[i], upper_bound[i], 20) for i in range(Ndim)])

fig = plt.figure(figsize=(40, 40))
grid = plt.GridSpec(Ndim, Ndim, wspace=0.1, hspace=0.1)


for i in range(Ndim):
    for j in range(i + 1):
        ax = fig.add_subplot(grid[i, j])
        if (i == 5 and j == 1) or (i == 1 and j == 0) or (i == 5 and j == 4):
            ax.spines["top"].set_linewidth(11)
            ax.spines["bottom"].set_linewidth(11)
            ax.spines["left"].set_linewidth(11)
            ax.spines["right"].set_linewidth(11)

fill_corner(
    fig,
    posterior_array[[0, 1, 4, 9, 5, 6, 7]][:, ::10],
    Ndim,
    axis,
    style=["--", "-"],
    fill_contours=False,
    lw=4,
    levels=[0.68, 0.95],
    smooth2d=0.1,
)

counter = 0
for i in range(Ndim):
    for j in range(i + 1):
        ax = fig.axes[counter]
        if i != Ndim - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(xlabel[j], fontsize=40)
            ax.set_xticklabels(np.around(ax.get_xticks(), 1), rotation=45, fontsize=30)
        if j != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(xlabel[i], fontsize=40)
            ax.set_yticklabels(np.around(ax.get_yticks(), 1), rotation=45, fontsize=30)

        counter += 1
        ax.tick_params(
            axis="both",
            direction="in",
            which="major",
            width=2.5,
            length=6,
            right=True,
            top=True,
        )
        ax.tick_params(
            axis="both",
            direction="in",
            which="minor",
            width=1.5,
            length=3,
            right=True,
            top=True,
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        if (i == j) and (i < 4):
            ax.patches[0].set_color("C0")
        elif (i == j) and (i >= 4):
            ax.patches[0].set_color("C1")
        elif (i < 4) and (j < 4):
            ax.collections[0].set_color("C0")
            ax.collections[0].set_facecolor("C0")
            ax.collections[0].set_alpha(0.5)
            ax.collections[1].set_color("C0")
            ax.collections[1].set_facecolor("C0")
            ax.collections[1].set_alpha(0.6)
        elif (i >= 4) and (j < 4):
            ax.collections[0].set_color("green")
            ax.collections[0].set_facecolor("green")
            ax.collections[0].set_alpha(0.5)
            ax.collections[1].set_color("green")
            ax.collections[1].set_facecolor("green")
            ax.collections[1].set_alpha(0.6)
        else:
            ax.collections[0].set_color("C1")
            ax.collections[0].set_facecolor("C1")
            ax.collections[0].set_alpha(0.5)
            ax.collections[1].set_color("C1")
            ax.collections[1].set_facecolor("C1")
            ax.collections[1].set_alpha(0.6)

ax2 = fig.add_axes([0.46, 0.69, 0.18, 0.18])
ax2.scatter(posterior_array[0], posterior_array[1], s=0.02, color="C0")
ax2.tick_params(
    axis="both",
    direction="in",
    which="major",
    width=2.5,
    length=6,
    right=True,
    top=True,
)
ax2.tick_params(
    axis="both",
    direction="in",
    which="minor",
    width=1.5,
    length=3,
    right=True,
    top=True,
)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.set_xlabel(xlabel[0], fontsize=40)
ax2.set_xticklabels(np.around(ax2.get_xticks(), 1), fontsize=30)
ax2.set_ylabel(xlabel[1], fontsize=40)
ax2.set_yticklabels(np.around(ax2.get_yticks(), 1), fontsize=30)


index1 = 1
index2 = 6
ax3 = fig.add_axes([0.69, 0.69, 0.18, 0.18])
ax3.scatter(posterior_array[index1], posterior_array[index2], s=0.02, color="green")
ax3.tick_params(
    axis="both",
    direction="in",
    which="major",
    width=2.5,
    length=6,
    right=True,
    top=True,
)
ax3.tick_params(
    axis="both",
    direction="in",
    which="minor",
    width=1.5,
    length=3,
    right=True,
    top=True,
)
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.set_xlabel(xlabel[1], fontsize=40)
ax3.set_xticklabels(np.around(ax3.get_xticks(), 1), fontsize=30)
ax3.set_ylabel(xlabel[5], fontsize=40)
ax3.set_yticklabels(np.around(ax3.get_yticks(), 1), fontsize=30)

ax4 = fig.add_axes([0.69, 0.465, 0.18, 0.18])
ax4.scatter(posterior_array[5], posterior_array[6], s=0.02, color="C1")
ax4.tick_params(
    axis="both",
    direction="in",
    which="major",
    width=2.5,
    length=6,
    right=True,
    top=True,
)
ax4.tick_params(
    axis="both",
    direction="in",
    which="minor",
    width=1.5,
    length=3,
    right=True,
    top=True,
)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())
ax4.set_xlabel(xlabel[4], fontsize=40)
ax4.set_xticklabels(np.around(ax4.get_xticks(), 1), fontsize=30)
ax4.set_ylabel(xlabel[5], fontsize=40)
ax4.set_yticklabels(np.around(ax4.get_yticks(), 1), fontsize=30)


custom_lines = [
    Line2D([0], [0], color="C0", lw=10),
    Line2D([0], [0], color="green", lw=10),
    Line2D([0], [0], color="C1", lw=10),
]

fig.legend(
    custom_lines,
    ["Progenitor Parameters", "Mixed", "Hyper-parameters"],
    loc=(0.72, 0.35),
    fontsize=40,
    frameon=False,
)

fig.savefig(
    paths.figures / "GW150914_corner_zoomed_lowres.png",
    bbox_inches="tight",
    dpi=100,
)

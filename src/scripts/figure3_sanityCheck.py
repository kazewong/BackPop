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

i = 0
data = h5py.File(paths.data / "GW150914_roots_raw.hdf5")
processed_data = h5py.File(paths.data / "GW150914_roots_processed.hdf5")
mcmc_processed = h5py.File(paths.data / "GWTC3_mcmc_processed.hdf5", "r")

m1_pred_mcmc = mcmc_processed["m1"][i]
m2_pred_mcmc = mcmc_processed["m2"][i]

m1_pred = processed_data["m1"]
m2_pred = processed_data["m2"]
t_merge = processed_data["t_merge"][()] / 1000

posterior_array = []
weight_array = []
observable_array = []
quantile_array = []

a, b, w, index = get_posterior(data["result_value"][0], data["result_coord"][0], 1e-1)
posterior_array.append(a)
weight_array.append(w)
observable_array.append(observables[i][index[0]])
quantile_array.append(np.quantile(a, [0.5 - 0.95 / 2, 0.5, 0.5 + 0.95 / 2], axis=0))
quantile_array = np.array(quantile_array)
plt.figure(figsize=(10, 9))
percentile = [0.68, 0.95]
corner.hist2d(
    m1_pred[()],
    m2_pred[()],
    weights=weight_array[0],
    color="C0",
    plot_density=False,
    labels="Root-finding",
    levels=percentile,
    plot_datapoints=False,
    contour_kwargs={
    "linewidths": 3,
    },
)
corner.hist2d(
    m1_pred_mcmc[()],
    m2_pred_mcmc[()],
    color="C1",
    plot_density=False,
    labels="MCMC",
    levels=percentile,
    plot_datapoints=False,
    contour_kwargs={
    "linewidths": 3,
    },
)
corner.hist2d(
    observables[i][:, 0],
    observables[i][:, 1],
    color="C2",
    plot_density=False,
    labels="Data",
    levels=percentile,
    plot_datapoints=False,
    contour_kwargs={
    "linewidths": 3,
    },
)
plt.xlabel(r"$M_1$", fontsize=30)
plt.ylabel(r"$M_2$", fontsize=30)
plt.title("GW150914")
custom_lines = [
    Line2D([0], [0], color="C0", lw=4),
    Line2D([0], [0], color="C1", lw=4),
    Line2D([0], [0], color="C2", lw=4),
]
plt.legend(custom_lines, ["Root-finding", "MCMC", "Data"], fontsize=20)
plt.savefig(
    paths.figures / "GW150914_reprojection.pdf",
    dpi=300,
)

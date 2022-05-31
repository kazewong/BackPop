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

posterior_array = []
data = h5py.File(paths.data / "GWTC3_mcmc_processed.hdf5", "r")
for i in range(data["coord_array"].shape[0]):
    evo_params = data["coord_array"][i][6]
    joint_dist = np.stack([data["m1"][i][()], evo_params], axis=-1)
    posterior_array.append(joint_dist)

posterior_array = np.array(posterior_array)
evo_params_array = np.delete(posterior_array, 29, axis=0)
fig, ax = plt.subplots(1, 1, figsize=(10, 9))

for i in range(data["coord_array"].shape[0]):
    try:
        if (
            i != 12
        ):  # Event number 12 has significant instability in KL divergence, that it is removed from the analysis.
            corner.hist2d(
                posterior_array[i, :, 0][posterior_array[i, :, 0] > 0],
                posterior_array[i, :, 1][posterior_array[i, :, 0] > 0],
                bins=80,
                smooth=0.5,
                label="GWTC3",
                plot_datapoints=False,
                plot_density=False,
                levels=[0.68],
                ax=ax,
                range=[[0, 45], [0, 1]],
                contour_kwargs={
                    "linewidths": 0,
                },
            )
            ax.collections[-2].set_color("C0")
            ax.collections[-2].set_alpha(0.1)
    except:
        continue
ax.set_xlabel(r"$m_{\rm 1}$")
ax.set_ylabel(r"$f_{\rm acc}$")
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
plt.savefig(
    paths.figures / "GWTC3_f_acc_mass.pdf",
    bbox_inches="tight",
)

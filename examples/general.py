# Tested using: https://compose.obspm.fr/eos/34

# %%
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import sys
sys.path.append("..")
from compose.general import EOS, EOSMetadata

# %%
md = EOSMetadata(
    pairs = {
        0: ("e", "electron"),
        10: ("n", "neutro"),
        11: ("p", "proton"),
        4002: ("He4", "alpha particle"),
        3002: ("He3", "helium 3"),
        3001: ("H3", "tritium"),
        2001: ("H2", "deuteron")
    },
    quads = {
        999: ("N", "average nucleous")
    }
)
eos = EOS("SFHo", md)
eos.compute_cs2()

# %%
eos.write("SFHo/compose.h5")

# %%
print("{} <= cs2 <= {}".format(eos.thermo["cs2"].min(), eos.thermo["cs2"].max()))

# %% Load comparison table
c = 29979245800.0           # CGS
MeV = 1.60217733e-06        # CGS
fm = 1e-13                  # CGS

ref_table = h5py.File("SFHo/SFHo_hydro_29-Jun-2015.h5", "r")
ref_t = np.array(ref_table["temperature"])
ref_ye = np.array(ref_table["ye"])
ref_mb = np.array(ref_table["mass_factor"])
ref_rho = np.array(ref_table["density"])
ref_cs2 = np.array(ref_table["cs2"])
ref_mb = ref_mb*MeV/c**2
ref_nb = ref_rho/(ref_mb)*fm**3

# %% Plot CompOSE EOS in pressure dominated regime
Y_e = 0.5
iyq = np.argmin(np.abs(eos.yq - Y_e))

plt.figure()
plt.plot(eos.t, eos.thermo["Q1"][0,iyq,:]/eos.t**4)
plt.xlabel(r"$T\ [{\rm MeV}]$")
plt.ylabel(r"$p/(nb*T^4)\ [{\rm MeV}^3]$")
plt.xscale("log")
plt.yscale("log")

# %% Compare CompOSE and stallarcollapse.org cs2 at zero density
Y_e = 0.5
iyq = np.argmin(np.abs(eos.yq - Y_e))
iye = np.argmin(np.abs(ref_ye - Y_e))

plt.figure()
plt.plot(eos.t, eos.thermo["cs2"][0,iyq,:], label="CompOSE")
plt.plot(ref_t[:], ref_cs2[iye,:,0], label="stellarcollapse.org")
plt.xlabel(r"$T\ [{\rm MeV}]$")
plt.ylabel(r"$c_s^2/c^2$")
plt.axhline(1/3, color='k', label=r"$1/3$")
plt.legend()
plt.xscale("log")
#plt.yscale("log")

# %% Compare CompOSE and stallarcollapse.org cs2 at zero temperature
Y_e = 0.5
iyq = np.argmin(np.abs(eos.yq - Y_e))
iye = np.argmin(np.abs(ref_ye - Y_e))

plt.figure()
plt.plot(eos.nb, eos.thermo["cs2"][:,iyq,0], label="CompOSE")
plt.plot(ref_nb, ref_cs2[iye,0,:], label="stellarcollapse.org")
plt.xlabel(r"$nb\ [{\rm fm}^{-3}]$")
plt.ylabel(r"$c_s^2/c^2$")
plt.axhline(1, color='k', label=r"$1$")
plt.legend()
plt.xscale("log")
plt.yscale("log")


# %% Plot sound speed for fixed Y_e
Y_e = 0.2
iyq = np.argmin(np.abs(eos.yq - Y_e))
iye = np.argmin(np.abs(ref_ye - Y_e))

plt.figure()
plt.title("CompOSE")
plt.pcolormesh(eos.nb, eos.t, eos.thermo["cs2"][:,iyq,:].T,vmin=0, vmax=1, cmap='jet')
plt.colorbar(label=r"$c_s^2$")
plt.xlabel(r"$n_b\ [{\rm fm}^{-3}]$")
plt.ylabel(r"$T\ [{\rm MeV}]$")
plt.xscale("log")
plt.yscale("log")

plt.figure()
plt.title("stellarcollapse.org")
plt.pcolormesh(ref_nb, ref_t, ref_cs2[iye,:,:],vmin=0, vmax=1, cmap='jet')
plt.colorbar(label=r"$c_s^2$")
plt.xlabel(r"$n_b\ [{\rm fm}^{-3}]$")
plt.ylabel(r"$T\ [{\rm MeV}]$")
plt.xscale("log")
plt.yscale("log")

# %%
plt.show()

# %%

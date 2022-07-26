# Tested using: https://compose.obspm.fr/eos/34

# %%
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
        2001: ("H2", "deuteron")},
    quads = {
        999: ("N", "average nucleous")
    }
)
eos = EOS("SFHo", md)

# %%
iyq = np.argmin(np.abs(eos.yq - 0.5))
print("Error on Y_e: ", np.max(np.abs(eos.Y["e"][:,iyq,:] - eos.yq[iyq])))

# %%
plt.figure()
plt.plot(eos.t, eos.thermo["Q1"][0,iyq,:]/eos.t**4)
plt.xlabel(r"$T\ [{\rm MeV}]$")
plt.ylabel(r"$p/(nb*T^4)\ [{\rm MeV^3}]$")
plt.xscale("log")
plt.yscale("log")

# %%
plt.figure()
plt.pcolormesh(eos.nb, eos.t, eos.Y["N"][:,iyq,:].T, norm=LogNorm(1e-15, 1.0))
plt.colorbar(label=r"$Y_{\rm heavy}$")
plt.xlabel(r"$n_b\ [{\rm fm}^{-3}]$")
plt.ylabel(r"$T\ [{\rm MeV}]$")
#plt.xscale("log")
plt.yscale("log")
plt.xlim(1e-11, 0.1)
plt.ylim(0.1, 100)

# %%
plt.show()

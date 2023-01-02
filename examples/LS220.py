# PyCompOSE: manages CompOSE tables
# Copyright (C) 2022, David Radice <david.radice@psu.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Tested using: https://compose.obspm.fr/eos/32
#
# This example shows how to import CompOSE tables and dump some data to ASCII
#
# Instructions
# - Create a folder "LS220" at the same location as this script
# - Download https://compose.obspm.fr/eos/32 and place it in a folder "LS220"
# - Run the script

# %%
# Import libraries
import os
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(os.path.join(SCRIPTDIR, os.pardir))
from compose.eos import Metadata, Table

# %%
# Read EOS table and restrict in region rho0 > 10^11 g/cm^3
md = Metadata(
    pairs = {
        0: ("e", "electron"),
        10: ("n", "neutro"),
        11: ("p", "proton"),
        4002: ("He4", "alpha particle"),
    },
)
eos = Table(md)
eos.read(os.path.join(SCRIPTDIR, "LS220"))
eos.restrict(nb_min=1e-4)

# %%
# Tabulated all indices of the table
idxs = []
for inb in range(eos.shape[0]):
    for iyq in range(eos.shape[1]):
        for it in range(eos.shape[2]):
            idxs.append((inb, iyq, it))
sel = idxs

# %%
# Output
#   rho0: rest mass density
#   Yq: charge fraction
#   e: total energy density
with open("ic_file.txt", "w") as ofile:
    for inb in range(eos.shape[0]):
        rho0 = Table.unit_dens*eos.nb[inb]*eos.mn
        for iyq in range(eos.shape[1]):
            yq = eos.yq[iyq]
            for it in range(eos.shape[2]):
                e = Table.unit_energy*eos.nb[inb]*eos.mn*(eos.thermo["Q7"][inb,iyq,it] + 1)
                ofile.write("{:e}\t{:e}\t{:e}\n".format(rho0, yq, e))
ofile.close()

with open("shape.txt", "w") as ofile:
    ofile.write("{} {} {}".format(*eos.shape))

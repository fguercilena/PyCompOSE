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

"""
Utilities to read general purpose (3D) EOS tables
"""

import h5py
import numpy as np
import os

class Metadata:
    """
    Class encoding the metadata/indexing used to read the EOS table

    Members

        thermo : list of extra quantities in the thermo table
        pairs  : dictionary of particle fractions in the compo table
        quad   : dictionary of isotope fractions in the compo table
    """
    def __init__(self, thermo=[], pairs={}, quads={}):
        """
        Initialize the metadata

        * thermo : list of additional (EOS specific) thermo quantities
        * pairs  : additional particles
        * quads  : additional isotopes

        All inputs are lists of tuples [(name, desc)]
        """
        self.thermo = {
            1: ("Q1", "pressure over number density: p/nb [MeV]"),
            2: ("Q2", "entropy per baryon [kb]"),
            3: ("Q3", "scaled and shifted baryon chemical potential: mu_b/m_n - 1"),
            4: ("Q4", "scaled charge chemical potential: mu_q/m_n"),
            5: ("Q5", "scaled effective lepton chemical potential: mu_l/m_n"),
            6: ("Q6", "scaled free energy per baryon: f/(nb*m_n) - 1"),
            7: ("Q7", "scaled internal energy per baryon: e/(nb*m_n) - 1")
        }
        for ix in range(len(thermo)):
            self.thermo[ix + 8] = thermo[ix]

        self.pairs = pairs.copy()
        self.quads = quads.copy()

class Table:
    """
    This class stores a table in CompOSE format.

    1D, 2D, and 3D tables are treated in the same way, with the only
    difference that some of the index ranges might be trivial.

    Data

        nb     : baryon number density [fm^-3]
        t      : temperature [MeV]
        yq     : charge fraction
        thermo : dictionary of 3D arrays containing the therm quantities
        Y      : dictionary of 3D arrays containing the number fractions
        A      : dictionary of 3D arrays containing the average mass of each isotope
        Z      : dictionary of 3D arrays containing the average charge of each isotope

    Metadata

        mn, mp : neutron and proton mass [MeV]
        lepton : if True, then leptons are included in the EOS

    The indexing for the 3D arrays is

        inb, iyq, it

    That is, the temperature is the fastest running index.
    """
    def __init__(self, metadata: Metadata, dtype=np.float64):
        """
        Initialize an EOS object

        * metadata : machine readable version of the EOS data sheet
        * dtype : data type
        """
        self.md = metadata
        self.dtype = dtype

        self.nb = np.empty(0)
        self.t = np.empty(0)
        self.yq = np.empty(0)
        self.shape = (self.nb.shape[0], self.yq.shape[0], self.t.shape[0])
        self.valid = np.ones(self.shape, dtype=bool)

        self.mn = np.nan
        self.mp = np.nan
        self.lepton = False

        self.thermo = {}
        self.Y, self.A, self.Z = {}, {}, {}

    def compute_cs2(self, floor=None):
        """
        Computes the square of the sound speed
        """
        P = self.thermo["Q1"]*self.nb[:,np.newaxis,np.newaxis]
        S = self.thermo["Q2"]
        u = self.mn*(self.thermo["Q7"] + 1)
        h = u + self.thermo["Q1"]

        dPdn = P*self.diff_wrt_nb(np.log(P))

        if self.t.shape[0] > 1:
            dPdt = P*self.diff_wrt_t(np.log(P))
            dSdn = S*self.diff_wrt_nb(np.log(S))
            dSdt = S*self.diff_wrt_t(np.log(S))

            self.thermo["cs2"] = (dPdn - dSdn/dSdt*dPdt)/h
        else:
            self.thermo["cs2"] = dPdn/h

        if floor is not None:
            self.thermo["cs2"] = np.maximum(self.thermo["cs2"], floor)
        self.md.thermo[12] = ("cs2", "sound speed squared [c^2]")

    def check_for_invalid_points(self, check_cs2_min=False, check_cs2_max=True):
        """
        Mark invalid points in the table
        """
        if check_cs2_min:
            self.valid = self.valid & (self.thermo["cs2"] > 0)
        if check_cs2_max:
            self.valid = self.valid & (self.thermo["cs2"] < 1)

    def diff_wrt_nb(self, Q):
        """
        Differentiate a 3D variable w.r.t nb

        This function is optimized for log spacing for nb, but will work with any spacing
        """
        log_nb = np.log(self.nb[:,np.newaxis,np.newaxis])
        dQdn = np.empty_like(Q)
        dQdn[1:-1,...] = (Q[2:,...] - Q[:-2,...])/(log_nb[2:] - log_nb[:-2])
        dQdn[0,...] = (Q[1,...] - Q[0,...])/(log_nb[1] - log_nb[0])
        dQdn[-1,...] = (Q[-1,...] - Q[-2,...])/(log_nb[-1] - log_nb[-2])
        return dQdn/self.nb[:,np.newaxis,np.newaxis]

    def diff_wrt_t(self, Q):
        """
        Differentiate a 3D variable w.r.t T

        This function is optimized for log spacing for T, but will work with any spacing

        NOTE: You will get an error if you try to differentiate w.r.t to T a 1D table
        """
        log_t = np.log(self.t[np.newaxis,np.newaxis,:])
        dQdt = np.empty_like(Q)
        dQdt[...,1:-1] = (Q[...,2:] - Q[...,:-2])/(log_t[...,2:] - log_t[...,:-2])
        dQdt[...,0] = (Q[...,1] - Q[...,0])/(log_t[0,0,1] - log_t[0,0,0])
        dQdt[...,-1] = (Q[...,-1] - Q[...,-2])/(log_t[0,0,-1] - log_t[0,0,-2])
        return dQdt/self.t[np.newaxis,np.newaxis,:]

    def restrict(self, nb_min=None, nb_max=None, yq_min=None, yq_max=None,
            t_min=None, t_max=None):
        """
        Restrict the table in the given range
        """
        if nb_min is not None:
            assert nb_min < self.nb[-1]
            in0 = self.nb.searchsorted(nb_min)
        else:
            in0 = None
        if nb_max is not None:
            in1 = self.nb.searchsorted(nb_max)
        else:
            in1 = None

        if yq_min is not None:
            assert yq_min < self.yq[-1]
            iy0 = self.yq.searchsorted(yq_min)
        else:
            iy0 = None
        if yq_max is not None:
            iy1 = self.yq.searchsorted(yq_max)
        else:
            iy1 = None

        if t_min is not None:
            assert t_min < self.t[-1]
            it0 = self.t.searchsorted(t_min)
        else:
            it0 = None
        if t_max is not None:
            it1 = self.t.searchsorted(t_max)
        else:
            it1 = None

        self.restrict_idx(in0, in1, iy0, iy1, it0, it1)

    def restrict_idx(self, in0=None, in1=None, iy0=None, iy1=None, it0=None, it1=None):
        """
        Restrict the table to a given indicial range
        """
        self.nb = self.nb[in0:in1]
        self.yq = self.yq[iy0:iy1]
        self.t = self.t[it0:it1]
        self.shape = (self.nb.shape[0], self.yq.shape[0], self.t.shape[0])
        self.valid = self.valid[in0:in1,iy0:iy1,it0:it1]

        for key in self.thermo.keys():
            self.thermo[key] = self.thermo[key][in0:in1,iy0:iy1,it0:it1]
        for key in self.Y.keys():
            self.Y[key] = self.Y[key][in0:in1,iy0:iy1,it0:it1]
        for key in self.A.keys():
            self.A[key] = self.A[key][in0:in1,iy0:iy1,it0:it1]
        for key in self.Z.keys():
            self.Z[key] = self.Z[key][in0:in1,iy0:iy1,it0:it1]

    def read(self, path):
        """
        Read the table from CompOSE ASCII format

        * path : folder containing the EOS in CompOSE format
        """
        self.path = path

        self.nb = np.loadtxt(os.path.join(path, "eos.nb"), skiprows=2, dtype=self.dtype)
        self.t = np.loadtxt(os.path.join(path, "eos.t"), skiprows=2, dtype=self.dtype).reshape(-1)
        self.yq = np.loadtxt(os.path.join(path, "eos.yq"), skiprows=2, dtype=self.dtype).reshape(-1)
        self.shape = (self.nb.shape[0], self.yq.shape[0], self.t.shape[0])
        self.valid = np.ones(self.shape, dtype=bool)

        L = open(os.path.join(path, "eos.thermo"), "r").readline().split()
        self.mn = float(L[0])
        self.mp = float(L[1])
        self.lepton = bool(L[2])

        self.__read_thermo_entries()
        self.__read_compo_entries()

    def __read_thermo_entries(self):
        """
        Parse eos.thermo using the given metadata key
        """
        self.thermo = {}
        for name, desc in self.md.thermo.values():
            self.thermo[name] = np.empty(self.shape, dtype=self.dtype)
        with open(os.path.join(self.path, "eos.thermo"), "r") as tfile:
            _ = tfile.readline()
            for line in tfile:
                L = line.split()
                it, inb, iyq = int(L[0])-1, int(L[1])-1, int(L[2])-1
                for iv in range(1, 8):
                    self.thermo[self.md.thermo[iv][0]][inb, iyq, it] = \
                        float(L[2 + iv])
                Nadd = int(L[10])
                for iv in range(8, 8+Nadd):
                    if iv in self.md.thermo:
                        self.thermo[self.md.thermo[iv][0]][inb, iyq, it] = \
                            float(L[2 + 1 + iv])

    def __read_compo_entries(self):
        """
        Parse eos.compo using the given metadata key
        """
        self.Y, self.A, self.Z = {}, {}, {}
        for name, desc in self.md.pairs.values():
            self.Y[name] = np.zeros(self.shape, dtype=self.dtype)
        for name, desc in self.md.quads.values():
            self.Y[name] = np.zeros(self.shape, dtype=self.dtype)
            self.A[name] = np.nan*np.ones(self.shape, dtype=self.dtype)
            self.Z[name] = np.nan*np.ones(self.shape, dtype=self.dtype)
        with open(os.path.join(self.path, "eos.compo"), "r") as cfile:
            for line in cfile:
                L = line.split()
                it, inb, iyq = int(L[0])-1, int(L[1])-1, int(L[2])-1
                Nphase = int(L[3])
                Npairs = int(L[4])
                ix = 5
                for ip in range(Npairs):
                    I, Y = int(L[ix]), float(L[ix + 1])
                    ix += 2
                    if I in self.md.pairs:
                        self.Y[self.md.pairs[I][0]][inb, iyq, it] = Y
                Nquad = int(L[ix])
                ix += 1
                for iq in range(Nquad):
                    I, A, Z, Y = int(L[ix]), float(L[ix+1]), \
                                 float(L[ix+2]), float(L[ix+3])
                    ix += 4
                    if I in self.md.quads:
                        name = self.md.quads[I][0]
                        self.A[name][inb, iyq, it] = A
                        self.Z[name][inb, iyq, it] = Z
                        self.Y[name][inb, iyq, it] = Y

    def shrink_to_valid_nb(self):
        """
        Restrict the range of nb
        """
        from .utils import find_valid_region

        if np.all(self.valid):
            return

        valid_nb = np.all(self.valid, axis=(1,2))
        in0, in1 = find_valid_region(valid_nb)

        self.restrict_idx(in0=in0, in1=in1)

    def write_hdf5(self, fname, dtype=np.float64):
        """
        Writes the table as an HDF5 file
        """
        dfile = h5py.File(fname, "w")
        dfile.create_dataset("nb", dtype=dtype, data=self.nb,
            compression="gzip", compression_opts=9)
        dfile.create_dataset("t", dtype=dtype, data=self.t,
            compression="gzip", compression_opts=9)
        dfile.create_dataset("yq", dtype=dtype, data=self.yq,
            compression="gzip", compression_opts=9)
        dfile["nb"].attrs["desc"] = "baryon number density [fm^-3]"
        dfile["t"].attrs["desc"] = "temperature [MeV]"
        dfile["yq"].attrs["desc"] = "charge fraction"

        dfile.create_dataset("mn", dtype=dtype, data=self.mn)
        dfile.create_dataset("mp", dtype=dtype, data=self.mp)
        dfile["mn"].attrs["desc"] = "neutron mass [MeV]"
        dfile["mp"].attrs["desc"] = "proton mass [MeV]"

        for name, desc in self.md.thermo.values():
            dfile.create_dataset(name, dtype=dtype, data=self.thermo[name],
                compression="gzip", compression_opts=9)
            dfile[name].attrs["desc"] = desc

        for name, desc in self.md.pairs.values():
            key = "Y[{}]".format(name)
            dfile.create_dataset(key, dtype=dtype, data=self.Y[name],
                compression="gzip", compression_opts=9)
            dfile[key].attrs["desc"] = desc

        for name, desc in self.md.quads.values():
            key = "Y[{}]".format(name)
            dfile.create_dataset(key, dtype=dtype, data=self.Y[name],
                compression="gzip", compression_opts=9)
            dfile[key].attrs["desc"] = desc

            key = "A[{}]".format(name)
            dfile.create_dataset(key, dtype=dtype, data=self.A[name],
                compression="gzip", compression_opts=9)
            dfile[key].attrs["desc"] = desc

            key = "Z[{}]".format(name)
            dfile.create_dataset(key, dtype=dtype, data=self.Z[name],
                compression="gzip", compression_opts=9)
            dfile[key].attrs["desc"] = desc

        dfile.close()
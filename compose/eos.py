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

from copy import deepcopy
import h5py
from math import pi
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

    """ multiply to convert MeV --> K """
    unit_temp  = 1.0/8.617333262e-11
    """ multiply to convert MeV/fm^3 --> g/cm^3 """
    unit_dens  = 1.782662696e12
    """ multiply to convert MeV/fm^3 --> dyn/cm^2 """
    unit_press = 1.602176634e33

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
        self.valid = np.zeros(self.shape, dtype=bool)

        self.mn = np.nan
        self.mp = np.nan
        self.lepton = False

        self.thermo = {}
        self.Y, self.A, self.Z = {}, {}, {}

    def copy(self, copy_data=True):
        """
        Returns a copy of the table

        * copy_data : if False, only the grid and metadata are copied
        """
        eos = Table(self.md, self.dtype)
        eos.nb = self.nb.copy()
        eos.t = self.t.copy()
        eos.yq = self.yq.copy()
        eos.shape = deepcopy(self.shape)
        eos.valid = self.valid.copy()
        eos.mn = self.mn
        eos.mp = self.mp
        eos.lepton = self.lepton

        if copy_data:
            for key, data in self.thermo.items():
                eos.thermo[key] = data.copy()
            for key, data in self.Y.items():
                eos.Y[key] = data.copy()
            for key, data in self.A.items():
                eos.A[key] = data.copy()
            for key, data in self.Z.items():
                eos.Z[key] = data.copy()

        return eos

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

    def eval_given_rtx(self, var, nb, yq, t):
        """
        Interpolates a given thermodynamic variable at the wanted locations

        * var : a 3D array with the data to interpolate
        * nb  : a 1D array of density points
        * t   : a 1D array of temperature points
        * yq  : a 1D array of charge fraction points

        NOTE: This is not meant to be particularly efficient
        """
        from scipy.interpolate import RegularGridInterpolator

        assert nb.shape == t.shape == yq.shape

        my_lnb = np.log(self.nb)
        my_lt = np.log(self.t)
        func = RegularGridInterpolator((my_lnb, self.yq, my_lt), var)

        xi = np.column_stack((np.log(nb).flatten(), yq.flatten(), np.log(t).flatten()))
        out = func(xi).reshape(nb.shape)

        return out

    def interpolate(self, nb_new, yq_new, t_new, method="cubic"):
        """
        Generate a new table by interpolating the EOS to the given grid

        * nb : 1D array with all the number densities
        * yq : 1D array with all the charge fractions
        * t  : 1D array with all the temperatures

        * method : interpolation method, is passed to scipy.RegularGridInterpolator

        NOTE: this only works for 3D tables
        """
        assert self.shape[0] > 1
        assert self.shape[1] > 1
        assert self.shape[2] > 1

        from scipy.interpolate import RegularGridInterpolator

        eos = Table(self.md, self.dtype)
        eos.nb = nb_new.copy()
        eos.t = t_new.copy()
        eos.yq = yq_new.copy()
        eos.shape = deepcopy((nb_new.shape[0], yq_new.shape[0], t_new.shape[0]))
        eos.valid = np.ones(eos.shape, dtype=bool)
        eos.mn = self.mn
        eos.mp = self.mp
        eos.lepton = self.lepton

        log_nb = np.log(self.nb)
        log_t = np.log(self.t)

        log_nb_new, yq_new, log_t_new = np.meshgrid(
                np.log(nb_new), yq_new, np.log(t_new), indexing='ij')
        xi = np.column_stack((log_nb_new.flatten(), yq_new.flatten(),
                log_t_new.flatten()))

        def interp_var_to_grid(var3d, log=False):
            if log:
                myvar = np.log(var3d)
            else:
                myvar = var3d
            func = RegularGridInterpolator((log_nb, self.yq, log_t),
                    myvar, method=method)
            res = func(xi).reshape(eos.shape)
            if log:
                return np.exp(res)
            return res

        for key in self.thermo.keys():
            if key == "Q1":
                eos.thermo[key] = interp_var_to_grid(self.thermo[key], True)
            else:
                eos.thermo[key] = interp_var_to_grid(self.thermo[key])
        for key in self.Y.keys():
            eos.Y[key] = interp_var_to_grid(self.Y[key])
        for key in self.A.keys():
            eos.A[key] = interp_var_to_grid(self.A[key])
        for key in self.Z.keys():
            eos.Z[key] = interp_var_to_grid(self.Z[key])

        return eos

    def make_beta_eq_table(self):
        """
        Create a new table in which yq is set by beta equilibrium

        Remark the new table will be invalidated
        """
        from .utils import find_beta_eq
        from .utils import interpolator

        def interp_to_given_yp(var3d, yq_eq):
            out = np.empty_like(yq_eq)
            for inb in range(var3d.shape[0]):
                for it in range(var3d.shape[2]):
                    f = interpolator(self.yq, var3d[inb,:,it])
                    out[inb,0,it] = f(yq_eq[inb,0,it])
            return out

        yq_eq = np.zeros((self.nb.shape[0], 1, self.t.shape[0]), dtype=self.dtype)
        for inb in range(len(self.nb)):
            for it in range(len(self.t)):
                # This is divided by the neutron mass, but it does not matter
                mu_l = self.thermo["Q5"]
                yq_eq[inb, 0, it] = find_beta_eq(self.yq, mu_l[inb, :, it])

        eos = self.copy(copy_data=False)
        eos.yq = np.zeros(1, dtype=self.dtype)
        eos.shape = (eos.nb.shape[0], 1, eos.t.shape[0])

        for key in self.thermo.keys():
            eos.thermo[key] = interp_to_given_yp(self.thermo[key], yq_eq)
        for key in self.Y.keys():
            eos.Y[key] = interp_to_given_yp(self.Y[key], yq_eq)
        for key in self.A.keys():
            eos.A[key] = interp_to_given_yp(self.A[key], yq_eq)
        for key in self.Z.keys():
            eos.Z[key] = interp_to_given_yp(self.Z[key], yq_eq)

        return eos

    def remove_photons(self):
        """
        Generate a new table without photons

        This takes care of removing photons from Q1, Q2, Q6, and Q7,
        but not from other quantities
        """
        nb = self.nb[:,np.newaxis,np.newaxis]
        t = self.t[np.newaxis,np.newaxis,:]

        # photon energy density [MeV fm^-3]
        e_ph = pi**2/15*t**4
        # photon pressure [MeV fm^-3]
        p_ph = 1/3*e_ph
        # photon free energy density [MeV fm^-3]
        f_ph = -p_ph
        # photon entropy density [fm^-3]
        s_ph = 4*pi**2/45*t**3

        eos = self.copy()

        p = self.thermo["Q1"]*nb
        eos.thermo["Q1"] = (p - p_ph)/nb

        s = self.thermo["Q2"]*nb
        eos.thermo["Q2"] = (s - s_ph)/nb

        f = self.mn*nb*(self.thermo["Q6"] + 1)
        eos.thermo["Q6"] = (f - f_ph)/(self.mn*nb)

        e = self.mn*nb*(self.thermo["Q7"] + 1)
        eos.thermo["Q7"] = (e - e_ph)/(self.mn*nb)

        return eos

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

    def slice_at_t_idx(self, it):
        """
        Constructs a new table at a fixed temperature self.t[it]
        """
        eos = self.copy(copy_data=False)
        eos.t = np.array(eos.t[it], dtype=self.dtype).reshape((-1))
        eos.shape = (eos.nb.shape[0], eos.yq.shape[0], 1)

        for key in self.thermo.keys():
            eos.thermo[key] = self.thermo[key][:,:,it].reshape(eos.shape)
        for key in self.Y.keys():
            eos.Y[key] = self.Y[key][:,:,it].reshape(eos.shape)
        for key in self.A.keys():
            eos.A[key] = self.A[key][:,:,it].reshape(eos.shape)
        for key in self.Z.keys():
            eos.Z[key] = self.Z[key][:,:,it].reshape(eos.shape)

        return eos

    def validate(self, check_cs2_min=False, check_cs2_max=True):
        """
        Mark invalid points in the table
        """
        self.valid[:] = True
        if check_cs2_min:
            self.valid = self.valid & (self.thermo["cs2"] > 0)
        if check_cs2_max:
            self.valid = self.valid & (self.thermo["cs2"] < 1)

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

    def write_lorene(self, fname):
        """
        Export the table in LORENE format. This is only possible for 1D tables.
        """
        assert self.shape[1] == 1
        assert self.shape[2] == 1

        with open(fname, "w") as f:
            f.write("#\n#\n#\n#\n#\n%d\n#\n#\n#\n" % len(self.nb))
            for i in range(len(self.nb)):
                nb = self.nb[i]
                e = Table.unit_dens*self.nb[i]*self.mn*(self.thermo["Q7"][i,0,0] + 1)
                p = Table.unit_press*self.thermo["Q1"][i,0,0]*self.nb[i]
                f.write("%d %.15e %.15e %.15e\n" % (i, nb, e, p))

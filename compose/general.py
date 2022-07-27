"""
Utilities to read general purpose (3D) EOS tables
"""

import h5py
import numpy as np
import os

class EOSMetadata:
    """
    Class encoding the metadata/indexing used to read the EOS table

    Members

        thermo : list of extra quantities in the thermo table
        pairs  : dictionary of particle fractions in the compo table
        quad   : dictionary of isotope fractions in the compo table
    """
    def __init__(self, thermo=[], pairs=[], quads=[]):
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

class EOS:
    """
    This class stores a three-dimensional table in format
    "General purpose EoS table" from CompOSE

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

    That is, the temperature is the fastest running index
    """
    def __init__(self, path, metadata: EOSMetadata, dtype=np.float64):
        """
        Initialize an EOS object

        * path : folder containing the EOS in CompOSE format
        """
        self.path = path
        self.md = metadata
        self.dtype = dtype

        self.nb = np.loadtxt(os.path.join(path, "eos.nb"), skiprows=2, dtype=dtype)
        self.t = np.loadtxt(os.path.join(path, "eos.t"), skiprows=2, dtype=dtype)
        self.yq = np.loadtxt(os.path.join(path, "eos.yq"), skiprows=2, dtype=dtype)
        self.shape = (self.nb.shape[0], self.yq.shape[0], self.t.shape[0])

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

    def diff_wrt_nb(self, Q):
        dQdn = np.empty_like(Q)
        dQdn[1:-1,...] = (Q[2:,...] - Q[:-2,...])/(
            self.nb[2:,np.newaxis,np.newaxis] - self.nb[:-2,np.newaxis,np.newaxis])
        dQdn[0,...] = (Q[1,...] - Q[0,...])/(self.nb[1] - self.nb[0])
        dQdn[-1,...] = (Q[-1,...] - Q[-2,...])/(self.nb[-1] - self.nb[-2])
        return dQdn

    def diff_wrt_t(self, Q):
        dQdt = np.empty_like(Q)
        dQdt[...,1:-1] = (Q[...,2:] - Q[...,:-2])/(
            self.t[np.newaxis,np.newaxis,2:] - self.t[np.newaxis,np.newaxis,:-2])
        dQdt[...,0] = (Q[...,1] - Q[...,0])/(self.t[1] - self.t[0])
        dQdt[...,-1] = (Q[...,-1] - Q[...,-2])/(self.t[-1] - self.t[-2])
        return dQdt

    def compute_cs2(self):
        """
        Computes the square of the sound speed
        """
        P = self.thermo["Q1"]*self.nb[:,np.newaxis,np.newaxis]
        S = self.thermo["Q2"]
        u = self.mn*(self.thermo["Q7"] + 1)
        h = u + self.thermo["Q1"] 

        dPdn = self.diff_wrt_nb(P)
        dSdn = self.diff_wrt_nb(S)
        dPdt = self.diff_wrt_t(P)
        dSdt = self.diff_wrt_t(S)

        self.thermo["cs2"] = (dPdn - dSdn/dSdt*dPdt)/h
        self.md.thermo[12] = ("cs2", "sound speed squared [c^2]")

    def write(self, fname, dtype=np.float64):
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
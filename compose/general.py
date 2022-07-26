"""
Utilities to read general purpose (3D) EOS tables
"""

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
    def __init__(self, path, metadata: EOSMetadata):
        """
        Initialize an EOS object

        * path : folder containing the EOS in CompOSE format
        """
        self.path = path
        self.md = metadata

        self.nb = np.loadtxt(os.path.join(path, "eos.nb"), skiprows=2)
        self.t = np.loadtxt(os.path.join(path, "eos.t"), skiprows=2)
        self.yq = np.loadtxt(os.path.join(path, "eos.yq"), skiprows=2)
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
            self.thermo[name] = np.empty(self.shape)
        with open(os.path.join(self.path, "eos.thermo"), "r") as tfile:
            _ = tfile.readline()
            for line in tfile:
                L = line.split()
                it, inb, iyq = int(L[0])-1, int(L[1])-1, int(L[2])-1
                for iv in range(1, 8):
                    self.thermo[self.md.thermo[iv][0]][inb, iyq, it] = float(L[2 + iv])
                Nadd = int(L[10])
                for iv in range(8, 8+Nadd):
                    if iv in self.md.thermo:
                        self.thermo[self.md.thermo[iv][0]][inb, iyq, it] = float(L[2 + 1 + iv])

    def __read_compo_entries(self):
        """
        Parse eos.compo using the given metadata key
        """
        self.Y, self.A, self.Z = {}, {}, {}
        for name, desc in self.md.pairs.values():
            self.Y[name] = np.zeros(self.shape)
        for name, desc in self.md.quads.values():
            self.Y[name] = np.zeros(self.shape)
            self.A[name] = np.nan*np.ones(self.shape)
            self.Z[name] = np.nan*np.ones(self.shape)
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
                    I, A, Z, Y = int(L[ix]), float(L[ix+1]), float(L[ix+2]), float(L[ix+3])
                    ix += 4
                    if I in self.md.quads:
                        name = self.md.quads[I][0]
                        self.A[name][inb, iyq, it] = A
                        self.Z[name][inb, iyq, it] = Z
                        self.Y[name][inb, iyq, it] = Y


def main():
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

if __name__ == '__main__':
    main()
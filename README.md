# PyCompOSE

A python library to read and manipulate equation of state tables in CompOSE
ASCII format. Currently supported functionality

* Read general 3D tables
* Compute sound speed
* Extract beta-equilibrated slices
* Export tables as HDF5 files

**Disclaimer.** This software is not part of the offical CompOSE project.
Use at your own risk.

## Usage

All CompOSE tables come with a data sheet listing quantities and conventions
used in the table. You will need to create a machine readable version of this
data sheet using the EOSMetadata class:

```python
from compose.eos import Metadata, Table

md = Metadata(
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
```

You do not have to list all particle species and their indices in the metadata,
but PyCompOSE will only read those quantities listed in the `EOSMetadata`.

Afterwards, you can read CompOSE tables as

```python
eos = Table(md)
eos.read("/path/to/my/eos")
```

The following code computes the sound speed, removes unphysical portions of the table, and outputs the table to HDF5

```python
eos.compute_cs2(floor=1e-6)
eos.check_for_invalid_points()
eos.shrink_to_valid_nb()
eos.write_hdf5(os.path.join(SCRIPTDIR, "SFHo", "compose.h5"))
```

All `EOS` quantities are stored in easily accessible `numpy` arrays:

```python
# Print range of cs^2
print("{} <= cs2 <= {}".format(eos.thermo["cs2"].min(), eos.thermo["cs2"].max()))
```

## License

Copyright (C) 2022, David Radice <david.radice@psu.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

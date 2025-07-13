"""
This is a helper file to read in Janaf nist tables and add them to the Janaf databse,
feel free to add your own. Just make sure you dont mix the gibbs free energies from
Janaf with others (or be very sure what you are doing).
"""

import re
import xarray as xr
import numpy as np
from urllib.request import urlopen, Request

# Links to the data, more species can be appended
species = {
    "H2": "https://janaf.nist.gov/tables/H-050.txt",
    "H2O": "https://janaf.nist.gov/tables/H-064.txt",
    "SiO": "https://janaf.nist.gov/tables/O-012.txt",
    "SiO2": "https://janaf.nist.gov/tables/O-040.txt",
    "SiO2[s]": "https://janaf.nist.gov/tables/O-037.txt",
    "C[s]": "",
    "C": ""
}

# this is the file to append to
file = 'janaf.nc'

# create dataset
ds = xr.Dataset(attrs=species)

for spec in species:
    # open link
    req = Request(
        url=species[spec],
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    lines = urlopen(req)

    # loop over each line
    i = 0
    outs = []
    for line in lines:
        # skip the first two entries
        if i < 2: i += 1; continue
        # make data into strings
        data = str(line)[2:-4].split(r'\t')
        # delete filler lines
        if 'TRANSITIO' in data or '<-->' in data[-1]:
            continue
        # add data
        outs.append((float(data[0]), float(data[-2])))

    # create xarray dataset of current species
    run = np.asarray(outs)
    da = xr.DataArray(
        data=run[:, 1],
        dims='temp_' + spec,
        coords={'temp_' + spec: run[:, 0]},
    )
    # save it to the dataset
    ds[spec] = da

# Save file
ds.to_netcdf(file)


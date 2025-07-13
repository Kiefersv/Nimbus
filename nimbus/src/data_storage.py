""" Functions to sotre and load data """
import xarray as xr

def save_run(self, sol, save_file=None, tag=None):

    # ==== How data is stored
    ds = xr.Dataset(
        data_vars={
            'qv': (['pressure'], sol.y[:self.sz, -1]),
            'qc': (['pressure'], sol.y[self.sz:self.sz * 2, -1]),
            'qn': (['pressure'], sol.y[self.sz * 2:self.sz * 3, -1]),
            'rg': (['pressure'], self.rg),
        },
        coords={
            'pressure': self.pres * 1e-6,
        },
        attrs={
            'mmw': self.mmw,
            'y': sol.y[:, -1],
            'itterations': self.loop_nr,
        },
    )

    # ==== store data in Nimbus class
    # define the tag
    if tag is None:
        tag = 'last_run'
    self.results[tag] = ds

    # ==== Print info
    print('[INFO] Saved run under tag: ' + tag)

    # ==== save data to file if a save file is given
    if not isinstance(save_file, type(None)):
        ds.to_netcdf(save_file + '.nc')
        print('       -> File name: ' + save_file)

    return ds



def load_previous_run(self, file_name, tag=None):
    """
    Load previously saved Nimbus runs.

    Parameters
    ----------
    self : Nimbus class
        current nimbus object.
    file_name : str
        Name of the file to load from working directory.
    tag : str, optional
        Name to store data in Nimbus.
    """

    # load file
    ds = xr.open_dataset(file_name)

    # if no tag is given use file name
    if tag is None:
        tag = file_name.split('.')[0]

    # load results into Nimbus
    self.results[tag] = ds

    # ==== Print info
    print('[INFO] Loaded previous run with tag: ' + tag)
    print('       -> File name: ' + file_name)

    # return results
    return ds
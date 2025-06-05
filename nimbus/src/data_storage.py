""" Functions to sotre and load data """
import xarray as xr

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

    # return results
    return ds
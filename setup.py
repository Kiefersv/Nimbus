""" set up file """
from setuptools import setup, find_packages

requirements = [
    "numpy",
    "scipy",
    "xarray",
    "matplotlib",
    "pandas",
    "astropy",
    "netcdf4",
    "h5netcdf",
]

setup(
    name='nimbus-exo',
    version='2.0.0',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    url='',
    author='Sven Kiefer',
    author_email='kiefersv.mail@gmail.com',
    description='Time dependent cloud model',
)


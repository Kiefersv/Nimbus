""" set up file """
from setuptools import setup, find_packages

# Read requirements from requirements.txt
def read_requirements():
    """ read requirements file """
    with open("requirements.txt") as req_file:
        return req_file.read().splitlines()

setup(
    name='nimbus',
    version='v2.0',
    packages=find_packages(),
    install_requires=read_requirements(),
    include_package_data=True,
    url='',
    author='Sven Kiefer',
    author_email='kiefersv.mail@gmail.com',
    description='Time dependent cloud model',
)

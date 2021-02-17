import os
from setuptools import setup, find_packages

BASEDIR = os.path.dirname(os.path.abspath(__file__))
VERSION = open(os.path.join(BASEDIR, 'VERSION')).read().strip()

BASE_DEPENDENCIES = [
    'wf-cv-datetime-utils>=0.1.0',
    'opencv-python>=4.5.1',
    'pandas>=1.2.2',
    'numpy>=1.20.1',
    'scipy>=1.6.0',
    'matplotlib>=3.3.4'
]

# allow setup.py to be run from any path
os.chdir(os.path.normpath(BASEDIR))

setup(
    name='wf-cv-utils',
    packages=find_packages(),
    version=VERSION,
    include_package_data=True,
    description='Miscellaneous utilities for working with camera data',
    long_description=open('README.md').read(),
    url='https://github.com/WildflowerSchools/wf-cv-utils',
    author='Theodore Quinn',
    author_email='ted.quinn@wildflowerschools.org',
    install_requires=BASE_DEPENDENCIES,
    keywords=['cv'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)

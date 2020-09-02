import os
from setuptools import setup, find_packages

BASEDIR = os.path.dirname(os.path.abspath(__file__))
VERSION = open(os.path.join(BASEDIR, 'VERSION')).read().strip()

BASE_DEPENDENCIES = [
    'wf-cv-datetime-utils>=0.1.0',
    'opencv-python>=3.4.1',
    'numpy>=1.14',
    'scipy>=1.1',
    'matplotlib>=2.2',
    'attrs>=19.3.0'
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

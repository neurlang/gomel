"""Setup script for phase package."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README_PYTHON.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from __init__.py
version = {}
with open(os.path.join(this_directory, '__init__.py')) as f:
    for line in f:
        if line.startswith('__version__'):
            exec(line, version)
            break

setup(
    name='phase-spectrogram',
    version=version.get('__version__', '0.0.1'),
    author='Neurlang Project',
    author_email='neurlang@proton.me',
    description='Phase-preserving spectrogram encoder/decoder for high-quality audio reconstruction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/neurlang/gomel',
    py_modules=['phase'],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'soundfile>=0.10.0',
        'Pillow>=8.0.0',
        'pypng>=0.20220715.0',
    ],
    extras_require={
        'dev': [
            'hypothesis>=6.0.0',
            'pytest>=6.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='audio spectrogram phase stft signal-processing',
)

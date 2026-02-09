"""
Setup script for nativity_analysis package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_path = Path(__file__).parent / 'requirements.txt'
with open(requirements_path, 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
readme_path = Path(__file__).parent / 'README.md'
with open(readme_path, 'r') as f:
    long_description = f.read()

setup(
    name='nativity_analysis',
    version='0.1.0',
    author='Dobbs Fertility Research Team',
    description='Bayesian analysis of birth patterns by nativity status',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/afranks86/dobbs_fertility',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Statistics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [
            'nativity-analyze=scripts.run_full_analysis:main',
            'nativity-figures=scripts.generate_figures:main',
        ],
    },
)

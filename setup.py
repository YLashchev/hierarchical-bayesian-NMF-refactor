"""Setup file for bayesian_panel_nmf package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bayesian_panel_nmf",
    version="0.1.0",
    author="Dobbs Fertility Research Team",
    description="Bayesian hierarchical panel models with low-rank factorization for causal inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/afranks86/dobbs_fertility",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpyro>=0.12.0",
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "pyyaml>=5.4",
        "joblib>=1.0.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "mypy>=0.900",
        ],
    },
    entry_points={
        "console_scripts": [
            "bpnmf-analyze=scripts.run_analysis:main",
        ],
    },
)

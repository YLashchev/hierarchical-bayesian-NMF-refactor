"""Setup file for bayesian_panel_nmf package."""

from setuptools import setup, find_packages

setup(
    name="bayesian_panel_nmf",
    version="0.1.0",
    description="Bayesian hierarchical panel models with low-rank factorization for causal inference",
    url="https://github.com/YLashchev/hierarchical-bayesian-NMF-refactor",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpyro>=0.12.0",
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "pyyaml>=5.4",
        "joblib>=1.0.0",
        "loguru>=0.6.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
    },
)

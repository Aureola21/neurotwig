from setuptools import setup, find_packages

setup(
    name="neurotwig",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "graphviz",
        "numpy",
        "matplotlib",
    ],
)
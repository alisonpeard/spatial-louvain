from setuptools import setup, find_packages

setup(
    name="spatial-louvain",
    version="0.15",
    author="Thomas Aynaud",
    author_email="taynaud@gmail.com",
    description="Louvain algorithm for community detection",
    license="BSD",
    url="https://github.com/taynaud/python-louvain",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Development Status :: 4 - Beta",
    ],

    packages=find_packages(),
    install_requires=[
        "networkx",
        "numpy",
        "scipy",
        "sklearn",
        "matplotlib"
    ],

    scripts=['bin/community']
)

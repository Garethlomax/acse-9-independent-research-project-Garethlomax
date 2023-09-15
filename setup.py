import setuptools

setuptools.setup(
    name="conflict_lstm",
    version="0.0.1",
    author="Gareth Lomax",
    author_email="gcl15@ic.ac.uk",
    description="Package for ConvLSTM analysis of conflict data",
    #    long_description=long_description,
    #    long_description_content_type="text/markdown",
    url="https://github.com/msc-acse/acse-9-independent-research-project-Garethlomax",
    packages=["conflict_lstm"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "sklearn",
        "matplotlib",
        "h5py",
        "pandas",
        "cartopy",
    ],
)

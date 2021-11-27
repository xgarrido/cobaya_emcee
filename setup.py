from setuptools import find_packages, setup

setup(
    name="cobaya_emcee",
    version="0.1",
    packages=find_packages(),
    description="A cobaya wrapper around emcee",
    url="https://github.com/xgarrido/cobaya_emcee.git",
    author="Xavier Garrido",
    author_email="xavier.garrido@ijclab.in2p3.fr",
    keywords=["cobaya", "MCMC", "emcee"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Education",
    ],
    install_requires=["emcee", "h5py", "schwimmbad"],
    package_data={"emcee": ["emcee.yaml"]},
    zip_safe=True,
)

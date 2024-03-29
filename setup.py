import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="apertools",
    version="0.7.0",
    author="Scott Staniewicz",
    author_email="scott.stanie@gmail.com",
    description="Tools for gathering and processing InSAR data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scottstanie/apertools",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: C",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    install_requires=[
        "numpy",
        # "scipy",
        # "mat4py",  # For loading matlab .mats
        "requests",
        "matplotlib",
        "click",
        "h5py",
        # "pillow",
        "sentineleof",
    ],
    entry_points={
        "console_scripts": [
            "aper=apertools.scripts.cli:cli",
            # "createdem=apertools.createdem:cli",
            "asfdownload=apertools.asfdownload:cli",
            "geocode=apertools.scripts.run_geocode:main",
        ],
    },
    zip_safe=False,
)

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="borg_parser",
    version="0.1.0",
    author="Jacob Kravits",
    author_email="jacob.kravits@colorado.edu",
    description="Library to parse borg runtime results",
    install_requires=[
        'more-itertools',
        'pandas',
        'numpy',
        'seaborn',
        'hiplot',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    include_package_data=True,
    package_data={'': ['example_runtime/water_energy.txt']},
)
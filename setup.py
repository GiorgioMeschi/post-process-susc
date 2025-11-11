import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="post_process_susc",
    version="1.0.0",
    author="Giorgio Meschi",
    author_email="giorgio.meschi@cimafoundation.org",
    description="Post process raw and tiled output from ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GiorgioMeschi/post-process-susc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # 'matplotlib',
        # 'numpy',
        # 'pandas',
        # 'geopandas',
        # 'rasterio',
        # 'scipy',
        # 'toolz',
        # 'scipy',
        # 'contextily'
      ],
)
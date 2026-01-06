# Laplacian Eigenmaps

This repository contains code to generate embeddings of the SDSS galaxy spectra dataset in lower dimensions using Laplacian Eigenmaps.


## Setup
Run if using a local conda environment:
```
conda install --file code/requirements.txt
```

or run using pip:
```
pip install -r code/requirements.txt
```
## Special Thanks
Thanks to the astroML database for providing the SDSS galaxy spectra database, https://github.com/mmp2/megaman/ for Riemannian metric estimation, https://github.com/yuchaz/independent_coordinate_search/ for Independent Eigendirection Selection, and for https://github.com/stephenportillo/SDSS-VAE/blob/master/SDSS-VAE.ipynb, from which the code of the dataset for the galaxy spectra was mostly adapted.

## Authors
Authors: Alexander Anthony Tang & Alexandre Gallet

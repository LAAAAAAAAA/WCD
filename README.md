# WCD: Diffraction-Aware Gravitational Wavelet Convolutional Detector

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

The WCD framework for lens gravitational wave recognition as described in:  
**"Identifying Microlensing by Compact Dark Matter through Diffraction Patterns in Gravitational Waves with Machine Learning"** ([arXiv:2509.04538](https://doi.org/10.48550/arXiv.2509.04538)).

## Overview
This project implements a Diffraction-Aware Gravitational Wavelet Convolutional Detector for lens gravitational wave recognition. 

## Key Features
- Integrates **multi-scale wavelet transforms** within residual convolutional blocks to enable diffraction-aware feature extraction, capturing intricate time-frequency interference patterns from wave-optics effects
- Employs astrophysically consistent training data incorporating realistic **dark matter mass distributions** and **redshift-dependent** lensing probabilities, ensuring physically meaningful feature learning
- Demonstrates machine learning-based adaptability across a wide range of lens masses without retraining or template adjustments, from stellar-mass to intermediate-mass dark matter objects
- Achieves larger effective receptive fields and superior parameter efficiency compared to standard convolutional architectures while maintaining high performance


## Installation

### Clone Repository
```bash
git clone https://github.com/LAAAAAAAAA/WCD.git
cd WCD
```

### Create Conda Environment
```bash
conda create -n wcd python=3.9.20
conda activate wcd
```

## Prerequisites
The GW image dataset used for training is available at:  
[https://doi.org/10.5281/zenodo.17038947](https://doi.org/10.5281/zenodo.17038947)  
Download and extract the dataset into the project directory.

## WCD Training and Evaluation

### For Original GW Dataset
1. Split the dataset into train/val/test sets
2. Place them in corresponding subdirectories under `workdir/splits/`:
   ```
   workdir/splits/
   ├── train/
        ├── 0
        └── 1
   ├── val/
        ├── 0
        └── 1
   └── test/
        ├── 0
        └── 1
   ```

### Training and Testing
```bash
python train.py
```

## Citation
If you find this project useful, please cite:[Identifying Microlensing by Compact Dark Matter through Diffraction Patterns in Gravitational Waves with Machine Learning](https://doi.org/10.48550/arXiv.2509.04538)

<!-- Associated paper: 
[Identifying Microlensing by Compact Dark Matter through Diffraction Patterns in Gravitational Waves with Machine Learning](https://doi.org/10.48550/arXiv.2509.04538)

Dataset: [10.5281/zenodo.15354648](https://doi.org/10.5281/zenodo.15354648) -->

## License
This project is licensed under the **BSD 3-Clause License** - see the [LICENSE](LICENSE) file for details.

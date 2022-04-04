# UC Berkeley MIDS W207 Applied Machine Learning Final Project
## Spring 2022

<br>

# Data Sources
Satellite images taken at 4 different bandwidths from the Copernicus Sentinel-2 mission stored on Microsoft Planetary Computer. In total, there are 11,748 chips, each with 5 items: 4 .tif images for each bandwidth and a label mask that details which pixels are cloud pixels and which are not cloud pixels.

# Project Organization

    ├── LICENSE
    ├── README.md                           <- The top-level README describing the project
    ├── notebooks                           <- Jupyter notebooks
        ├── archive                         <- Old code from preivous iterations of notebooks
        ├── benchmark_src                   <- Python files returned from running a model
        └── lightning_logs                  <- Logs resulting from PyTorch Lighting
    ├── references                          
        ├── planetary-computer-containers   <- Pre-loaded containers from Planetary Computer
        └── identify_mis_labeled_chips      <- Code, PNGs, CSVs used to find bad chips


# Environment

All files run in Microsoft Planetary Computer when `PyTorch - GPU` option is selected.
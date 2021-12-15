## Installation

Firstly, clone this repository.

```bash
git clone https://github.com/obrookes/PanAfrican-Ape-Action-Recognition.git
cd PanAfrican-Ape-Action-Recognition
```

Install [Anaconda](https://docs.conda.io/en/latest/miniconda.html). Then create a conda environment using the environment file (environment.yml) using the following command.

```bash
conda env create -f environment.yml
```

Activate this conda environment:

```bash
conda activate action
```

This should install the requisite packages.


## Usage

A quick start script for training on CPU is provided below:

```bash
python train.py --compute='local'
                --gpus=0
                --nodes=0
                --batch_size=8
                --num_workers=12
                --freeze_backbone=0
                --epochs=10
```

If you want to train on GPU (across multiple GPUs, across multiple nodes) use the following:

```bash
python train.py --compute='hpc'
                --gpus=2 
                --nodes=1 
                --batch_size=8 
                --balanced_sampling='dynamic'
                --num_workers=12
                --freeze_backbone=0
                --epochs=10
```

This has been tested on SLURM only.



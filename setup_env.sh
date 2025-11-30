#!/bin/bash

module load anaconda
conda create -n ssl_env python=3.10 -y
conda activate ssl_env

pip install -r requirements.txt

echo "Environment setup complete."

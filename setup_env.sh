#!/usr/bin/env bash

#### VARIABLES TO DEFINE ####
# lynx-id project location on Jean Zay
export LYNX_PROJECT=$WORK/DP-SCR_Identify-and-estimate-density-lynx-population

#### ACTIVATE ENVIRONMENT ####
# Load Jean Zay module
module purge
module load pytorch-gpu/py3/2.0.1

# Path to the directory containing dev tool installations
export PYTHONUSERBASE=$LYNX_PROJECT/.local
export PATH=$LYNX_PROJECT/.local/bin:$PATH

# Install Python packages specified in the requirements.txt file of MegaDetector
# pip install -r $LYNX_PROJECT/data_pipeline/megadetector/MegaDetector/envs/requirements.txt --user --no-cache-dir

# Add MegaDetector directory to the Python path
export PYTHONPATH="$PYTHONPATH:LYNX_PROJECT/data_pipeline/megadetector/MegaDetector"

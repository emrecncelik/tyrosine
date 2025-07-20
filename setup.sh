#!/bin/bash
set -e

echo "🐍 Creating new conda environment 'net2brain' with Python 3.11..."
conda create -n net2brain python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate net2brain    

echo "📦 Installing required packages..."
pip install -U git+https://github.com/cvai-roig-lab/Net2Brain
pip install -r requirements.txt

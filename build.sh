#!/usr/bin/env bash
# exit on error
set -o errexit

# Install CPU-only PyTorch first to drastically reduce build size
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
pip install -r requirements.txt

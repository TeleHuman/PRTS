#!/bin/bash
which python

set -e 

echo "Installing fused_rmsnorm..."
python rms_setup.py install

echo "fused_rmsnorm installations complete."
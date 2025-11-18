#!/bin/bash

# Upgrade pip and install PyTorch and its dependencies first
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.1.2+cpu torchvision==0.16.2+cpu

# Now, install the rest of the requirements, including building detectron2
# We use --no-cache-dir to ensure a clean build
pip install --no-cache-dir -r requirements.txt

# Run the streamlit app
streamlit run app.py

#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Execute Python scripts
cd preprocessing
python ./preprocessing.py
cd ..
cd training
python ./train.py
cd ..
cd evaluation
python ./eval.py
cd ..

# Deactivate the virtual environment
deactivate

# Build the Docker image
docker build --build-arg EXPERIMENT_ID=737694674622074143 --build-arg RUN_ID=c09a656b6f624a72b6897ad6dcb7c122 -t my-flask-app .

# Run Docker container
docker run -p 5000:5000 -d my-flask-app

echo "All done!"
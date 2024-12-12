# Hierarchical Autoencoder for CMS L1 Trigger Data Compression

This project implements a hierarchical convolutional autoencoder network designed to compress particle detector data from the CMS experiment while preserving physics-motivated quantities. The network is structured to mirror the hierarchical nature of the CMS detector's trigger system, operating at multiple stages (ROC, Module, Triplet, Stage1, Stage2). It has been developed by Pradyun Hebbar (LBNL), Andre David (CERN), Thorben Quast (CERN) and Don Winter (RWTH Aachen).

## Overview

The autoencoder network is designed to:
- Compress particle detector data while maintaining physics-relevant information
- Extract important physics quantities like particle energy and momentum
- Operate across multiple trigger stages of the CMS detector
- Handle data from GEANT4 simulations as proof of concept

## Requirements

- Python 3.x
- TensorFlow
- QKeras (v0.8 or later)
- NumPy
- Matplotlib
- imageio
- pickle

## Network Architecture

The network consists of five hierarchical levels:

1. **ROC Level**: Processes raw detector data
   - Input shape: (4, 4, 1)
   - Uses quantized convolution layers
   - Includes flattening and dense layers

2. **Module Level**: Processes ROC-level encoded data
   - Input shape: (3, latent_dim_ROC)
   - Uses quantized dense layers

3. **Triplet Level**: Processes module-level encoded data
   - Input shape: (3, latent_dim_MODULE)
   - Uses quantized dense layers

4. **Stage1 Level**: Processes triplet-level encoded data
   - Input shape: (19, latent_dim_TRIPLET)
   - Uses quantized dense layers

5. **Stage2 Level**: Processes stage1-level encoded data
   - Input shape: (30, latent_dim_STAGE1)
   - Uses quantized dense layers

## Training Process

The training follows a hierarchical approach:
1. Initial training of all levels (epochs 1-4)
2. Fine-tuning of ROC level (epochs 5-200)
3. Fine-tuning of Module level (epochs 201-500)
4. Fine-tuning of Triplet level (epochs 501-900)
5. Fine-tuning of Stage1 level (epochs 901-1400)
6. Fine-tuning of Stage2 level (epochs 1401+)

## Usage

### Basic Usage

```bash
python train_experimental_cnn.py
```

### Command Line Arguments

- `--dataTrain`: Path to training data pickle file
- `--costFigure`: Path for cost function evolution plot
- `--animFile`: Path for animation file
- `--modelFile`: Path for model saving/loading
- `--batchSize`: Training batch size (default: 2)
- `--NEpochs`: Number of training epochs (default: 2000)
- `--LatentDimension`: Dimension of latent space (default: 8)
- `--GPU`: GPU device to use (default: 3)

### Example

```bash
python train_experimental_cnn.py \
    --dataTrain /path/to/data/electrons_config1_1_to_500GeV_4Tesla.pickle \
    --costFigure /path/to/output/cost.pdf \
    --modelFile /path/to/output/model \
    --batchSize 32 \
    --NEpochs 2000 \
    --LatentDimension 16 \
    --GPU 0
```

## Output and Visualization

The training process generates:
- Cost function evolution plots
- Visualization of input vs. reconstructed detector data
- Model checkpoints at each epoch
- Training progress metrics

## Model Features

- Uses quantized layers for efficient implementation
- Implements both encoding and decoding pathways
- Supports recursive encoding/decoding across hierarchical levels
- Includes visualization tools for monitoring training progress
- Saves/loads model state and training progress

## Data Format

Input data should be provided as a pickle file containing:
- `digi`: Detector digitization data
- `meta`: Associated metadata

## Authors

Pradyun Hebbar (LBNL), Andre David (CERN), Thorben Quast (CERN) and Don Winter (RWTH Aachen).

# MNIST Classifier

A simple PyTorch implementation of a Multi-Layer Perceptron (MLP) for classifying MNIST handwritten digits. This project provides a modular structure for training and evaluating neural networks on the MNIST dataset.

## Features

- Configurable MLP architecture with customizable:
  - Number of layers
  - Hidden layer dimensions
  - Activation functions (ReLU, Tanh, Sigmoid)
  - Dropout rate
- Flexible training configuration:
  - Multiple optimizers (Adam, SGD)
  - Various loss functions (Cross Entropy, MSE)
  - Customizable batch size and number of workers
- Experiment tracking:
  - Automatic logging of training and test losses
  - Loss visualization plots
  - Configurable experiment naming and results storage

## Project Structure

```
mnist/
├── configs/              # Configuration files
├── data.py              # Data loading and preprocessing
├── model.py             # MLP model implementation
├── train.py             # Training loop and utilities
├── utils.py             # Helper functions
├── config.py            # Configuration management
└── run.py              # Main execution script
```

## Prerequisites
    - Python 3.x
    - PyTorch
    - NumPy
    - Matplotlib
    - PyYAML
    - idx2numpy (for MNIST data loading)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/joshuaplacidi/mnist.git
cd mnist
```

2. Install dependencies:
```bash
pip -r requirements.txt
```

3. Download MNIST dataset from Kaggle :

    - Go to https://www.kaggle.com/datasets/hojjatk/mnist-dataset
    - Click Download > Download dataset as zip
    - Extract the file, this should give you
        - train-images-idx3-ubyte (train)
        - train-labels-idx1-ubyte (train)
        - t10k-images-idx3-ubyte (test)
        - t10k-labels-idx1-ubyte (test)

## Usage

1. Configure your experiment:
    - Edit `configs/mlp.yaml` to set your desired:
        - Model architecture
        - Training parameters
        - Data paths
        - Experiment settings

2. Run the training:
```bash
python run.py --config configs/mlp.yaml
```

3. View results:
    - Training logs are saved in the specified results directory
    - Loss plots are automatically generated as `losses.png`
    - Check the logs.csv file for detailed training metrics

## Configuration

The configuration file (`config.yaml`) allows you to customize:

- Model architecture:
    - Input dimension
    - Hidden layer dimension
    - Number of layers
    - Activation function
    - Dropout rate

- Training parameters:
    - Number of epochs
    - Learning rate
    - Optimizer choice
    - Loss function
    - Batch size

- Experiment settings:
    - Experiment name
    - Results directory
    - Device (CPU/GPU)

## Results

After training, you'll find:
- `logs.csv`: Detailed training metrics
- `losses.png`: Visualization of training and test losses
- Model checkpoints (if configured)

## Author

Joshua Placidi
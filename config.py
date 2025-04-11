"""
Configuration management for MNIST classifier experiments.

This module provides functionality to load and validate YAML configuration files
for MNIST classifier experiments. It ensures that all required parameters are
present and have the correct types and values.

Author: Joshua Placidi
"""

import yaml
from typing import Dict, Any
import os

def validate_data_config(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the data configuration.
    """
    required_fields = {
        'train_images_path': str,
        'train_labels_path': str,
        'test_images_path': str,
        'test_labels_path': str,
        'batch_size': int,
        'num_workers': int
    }
    
    for field, field_type in required_fields.items():
        if field not in data_config:
            raise ValueError(f"Missing required field '{field}' in data config")
        if not isinstance(data_config[field], field_type):
            raise ValueError(f"Field '{field}' must be of type {field_type.__name__}")

    # Expand user paths
    for field, value in data_config.items():
        if field.endswith('_path') and value.startswith('~'):
            data_config[field] = os.path.expanduser(value)

    return data_config


def validate_model_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the model configuration.
    """
    required_fields = {
        'input_dim': int,
        'hidden_dim': int,
        'output_dim': int,
        'num_layers': int,
        'activation_fn': str,
        'dropout': float
    }
    
    for field, field_type in required_fields.items():
        if field not in model_config:
            raise ValueError(f"Missing required field '{field}' in model config")
        if not isinstance(model_config[field], field_type):
            raise ValueError(f"Field '{field}' must be of type {field_type.__name__}")
            
    valid_activations = ['relu', 'tanh', 'sigmoid']
    if model_config['activation_fn'] not in valid_activations:
        raise ValueError(f"activation_fn must be one of {valid_activations}")

    return model_config


def validate_training_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the training configuration.
    """
    required_fields = {
        'num_epochs': int,
        'learning_rate': float,
        'optimizer': str,
        'criterion': str
    }
    
    for field, field_type in required_fields.items():
        if field not in training_config:
            raise ValueError(f"Missing required field '{field}' in training config")
        if not isinstance(training_config[field], field_type):
            raise ValueError(f"Field '{field}' must be of type {field_type.__name__}")
            
    valid_optimizers = ['adam', 'sgd']
    if training_config['optimizer'] not in valid_optimizers:
        raise ValueError(f"optimizer must be one of {valid_optimizers}")
        
    valid_criteria = ['cross_entropy', 'mse']
    if training_config['criterion'] not in valid_criteria:
        raise ValueError(f"criterion must be one of {valid_criteria}")
        
    if 'optimizer_params' not in training_config:
        raise ValueError("Missing optimizer_params in training config")
        
    required_optimizer_params = {
        'weight_decay': float,
        'beta1': float,
        'beta2': float
    }
    
    for param, param_type in required_optimizer_params.items():
        if param not in training_config['optimizer_params']:
            raise ValueError(f"Missing required parameter '{param}' in optimizer_params")
        if not isinstance(training_config['optimizer_params'][param], param_type):
            raise ValueError(f"Parameter '{param}' must be of type {param_type.__name__}")

    return training_config


def validate_experiment_config(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the experiment configuration.
    """
    required_fields = {
        'name': str,
        'save_dir': str,
    }
    
    for field, field_type in required_fields.items():
        if field not in experiment_config:
            raise ValueError(f"Missing required field '{field}' in experiment config")
        if not isinstance(experiment_config[field], field_type):
            raise ValueError(f"Field '{field}' must be of type {field_type.__name__}")

    return experiment_config


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate the configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check main sections exist
    required_sections = ['data', 'model', 'training', 'experiment', 'device']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config")

    # Validate each section
    config['data'] = validate_data_config(config['data'])
    config['model'] = validate_model_config(config['model'])
    config['training'] = validate_training_config(config['training'])
    config['experiment'] = validate_experiment_config(config['experiment'])

    # Validate device
    if not (config['device'] == 'cpu' or config['device'].startswith('cuda:')):
        raise ValueError("device must be either 'cpu' or start with 'cuda:' (e.g. 'cuda:0')")

    return config


if __name__ == "__main__":
    config = load_config("configs/config.yaml")
    print(config)

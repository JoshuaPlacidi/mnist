import torch
from config import load_config
from model import MLP
from data import get_dataloaders
from train import Trainer

def run(config_path: str):

    config = load_config(config_path)

    # Get dataloaders
    train_loader, test_loader = get_dataloaders(
        train_images_path=config['data']['train_images_path'],
        train_labels_path=config['data']['train_labels_path'],
        test_images_path=config['data']['test_images_path'],
        test_labels_path=config['data']['test_labels_path'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
    )
    
    # Get model
    model = MLP(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        num_layers=config['model']['num_layers'],
        activation_fn=config['model']['activation_fn'],
        dropout=config['model']['dropout'],
    )

    # Get trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=config['training']['optimizer'],
        criterion=config['training']['criterion'],
        num_epochs=config['training']['num_epochs'],
        results_dir=config['experiment']['save_dir'],
    )

    # Train model
    trainer.train()
    trainer.plot_losses()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    run(args.config)


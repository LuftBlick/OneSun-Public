"""
Training functions for models.
"""
import torch
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.initialization import get_default_training_config, init_models, init_optimizers
from utils.daily_dataset import load_and_split_dataset
from utils.trainer import ModelTrainer
from utils.loss_functions import combined_loss
from utils.model_eval import ModelEvaluator
from utils.dataset_manager import DatasetManager


data_path = 'G:/.shortcut-targets-by-id/0B9RCVyuhAWswTW0zRUtGcGJEdjQ/staff/datasharing/AK/OneSun/data/l0_datasets/'
fn = 'Pandora117s1_Rome-SAP_20240701_20241231.pkl'

station_params = {
    'latitude': 41.9028,
    'longitude': 12.4964,
    'altitude': 21
}

def setup_training(training_config=None):
    """
    Set up the training environment using a pre-created dataset
    """
    if training_config is None:
        training_config = get_default_training_config()
    
    # Initialize dataset manager and load datasets
    dataset_manager = DatasetManager()
    dataset_id = training_config['dataset_name']

    
    # Load datasets
    train_dataset = dataset_manager.load_dataset(dataset_id, dataset_type='train')
    val_dataset = dataset_manager.load_dataset(dataset_id, dataset_type='val')
    
    if train_dataset is None or val_dataset is None:
        raise ValueError(f"Dataset '{dataset_id}' not found. Please create it first using the dataset manager.")
    
    # Create data loaders
    print(f"Creating data loaders with batch size {training_config['batch_size']}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False
    )
    
    # Get a sample for model initialization
    sample = train_dataset[0]
    
    # Initialize models & optimizers
    atmospheric_model = training_config.get('atmospheric_model', 'Atmospheric Model')
    instrument_model = training_config.get('instrument_model', 'Instrument Model')
    solar_conv = training_config.get('solar_conv')
    gas_ods = training_config.get('gas_ods')

    model_i, model_a = init_models(sample,
                                   atmospheric_model=atmospheric_model,
                                   instrument_model=instrument_model, gas_ods=gas_ods, solar_conv=solar_conv)
    optimizer_i, optimizer_a = init_optimizers(model_i, model_a, training_config)
    
    # Setup trainer
    trainer = ModelTrainer(
        model_i=model_i,
        model_a=model_a,
        experiment_name=training_config['experiment_name'],
        save=training_config['save'],
        dataset_name=dataset_id,  # Use dataset ID as reference
        mse_loss_w=training_config['mse_loss_w'],
        corr_loss_w=training_config['corr_loss_w'],
        sam_loss_w=training_config['sam_loss_w'],
        cos_loss_w=training_config['cos_loss_w']
    )
    
    # Save configuration with dataset reference
    trainer.save_config(training_config)
    
    return {
        'config': training_config,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'model_i': model_i, 
        'model_a': model_a,
        'optimizer_i': optimizer_i,
        'optimizer_a': optimizer_a,
        'trainer': trainer,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'dataset_id': dataset_id
    }


def train(components=None, config=None, training_status=None):
    """
    Run the training process with validation.
    
    Args:
        components: Dictionary of training components from setup_training()
        config: Optional configuration dictionary
    
    Returns:
        Dictionary with training results and components
    """
    if components is None:
        components = setup_training(config)
    
    config = components['config']
    trainer = components['trainer']
    train_loader = components['train_loader']
    val_loader = components['val_loader']
    model_i = components['model_i']
    model_a = components['model_a']
    optimizer_i = components['optimizer_i']
    optimizer_a = components['optimizer_a']
    
    # Create evaluator
    evaluator = ModelEvaluator(trainer=trainer)
    
    # Training loop
    train_losses = []
    valid_losses = []
    early_stopped = False
    start_time = time.time()
    
    try:
        for epoch in range(config['epochs']):
            epoch_start = time.time()
            print(f"Epoch {epoch+1}/{config['epochs']}")
            
            # Train
            train_loss = trainer.train_epoch(
                data_loader=train_loader, 
                optimizer_i=optimizer_i, 
                optimizer_a=optimizer_a, 
                epoch=epoch
            )
            train_losses.append(train_loss)
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validation
            _, val_loss = evaluator.evaluate_models(
                model_i, 
                model_a, 
                val_loader, 
            )
            valid_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.4f}")
            if training_status:
                training_status.info(f"Epoch {epoch+1}/{config['epochs']};Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            epoch_time = time.time() - epoch_start
            print(f"Epoch completed in {epoch_time:.2f} seconds")
            
            # Save checkpoint with both losses
            if config.get('save', True):
                checkpoint_data = {
                    'epoch': epoch,
                    'model_i_state_dict': model_i.state_dict(),
                    'model_a_state_dict': model_a.state_dict(),
                    'optimizer_i_state_dict': optimizer_i.state_dict(),
                    'optimizer_a_state_dict': optimizer_a.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_losses': train_losses,
                    'valid_losses': valid_losses
                }
                trainer.save_checkpoint_data(checkpoint_data)
            
            # Optional early stopping
            if config.get('early_stopping', False) and len(valid_losses) > config.get('patience', 5):
                # Check if validation loss hasn't improved for 'patience' epochs
                patience = config.get('patience', 5)
                if val_loss > min(valid_losses[:-patience]):
                    print(f"Early stopping triggered: no improvement for {patience} epochs")
                    early_stopped = True
                    break
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        # Calculate training time
        training_time = time.time() - start_time
        
        # Save final models
        if config.get('save', True):
            trainer.save_final_models()
            
            # Save comprehensive experiment metrics
            metrics_path = trainer.run_dir / "experiment_metrics.json"
            metrics = {
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'best_val_loss': min(valid_losses) if valid_losses else None,
                'best_epoch': valid_losses.index(min(valid_losses)) + 1 if valid_losses else None,
                'training_time': training_time,
                'epochs_completed': len(train_losses),
                'early_stopped': early_stopped,
                'final_lr_i': optimizer_i.param_groups[0]['lr'],
                'final_lr_a': optimizer_a.param_groups[0]['lr'],
                'mean_epoch_time': training_time / len(train_losses) if train_losses else 0,
                'train_val_loss_ratio': train_losses[-1] / valid_losses[-1] if train_losses and valid_losses else None
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4, default=lambda o: float(o) if isinstance(o, np.float32) else o)
        
        # Plot loss curves
        if len(train_losses) > 0:
            fig = evaluator.plot_loss_curves(train_losses, valid_losses)
            if config.get('save', True):
                fig_path = trainer.run_dir / "loss_curves.png"
                fig.savefig(fig_path)
                #plt.close(fig)
        
        print(f"Training completed in {training_time:.2f} seconds!")
        if early_stopped:
            print("Training was early stopped")
        print(f"Best validation loss: {min(valid_losses) if valid_losses else 'N/A'}")
        print(f"Best epoch: {valid_losses.index(min(valid_losses)) + 1 if valid_losses else 'N/A'}")
        
        trainer.close()
    
    return {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'components': components,
        'training_time': training_time,
        'early_stopped': early_stopped
    }

# Still keep the main function for command-line use
def main():
    train()

if __name__ == "__main__":
    
    main()
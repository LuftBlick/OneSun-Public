import torch
import torch.nn as nn
from datetime import datetime
import os
import json
from pathlib import Path

from mpmath.libmp.libmpc import beta_crossover
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from utils.loss_functions import combined_loss, spectral_fidelity_loss, daily_correlation_loss

class ModelTrainer:
    def __init__(self, model_i, model_a, experiment_name="default", keep_best_n=3, save=True, dataset_name="unknown",mse_loss_w=0.1, corr_loss_w=0.1, sam_loss_w=0.1, cos_loss_w=0.1):
        self.model_i = model_i
        self.model_a = model_a
        self.experiment_name = experiment_name
        self.keep_best_n = keep_best_n
        self.save = save
        self.dataset_name = dataset_name
        self.mse_loss_w = mse_loss_w
        self.corr_loss_w = corr_loss_w
        self.sam_loss_w = sam_loss_w
        self.cos_loss_w = cos_loss_w
        self.best_losses = []  # Keep track of best checkpoints
        env_exp_runs = os.getenv("EXPERIMENT_RUNS_DIR")
        if env_exp_runs:
            base_dir = Path(env_exp_runs)
        else:
            base_dir = Path(__file__).resolve().parents[2] / "experiment_runs"
        #base_dir= Path(save_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = base_dir / f"{experiment_name}_{timestamp}"
        
        self.models_dir = self.run_dir / "models"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Setup directories, with timestamp only if save=True
        #base_dir = Path.cwd().parent / "experiment_runs"
        base_dir.mkdir(parents=True, exist_ok=True)
        print(f"DEBUG: ModelTrainer base_dir: {base_dir}")
        if self.save:
            # Create unique directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = base_dir / f"{experiment_name}_{timestamp}"
        else:
            # Use a fixed "latest" directory that gets overwritten
            self.run_dir = base_dir / f"{experiment_name}_latest"
            # Clean directory if it exists
            if self.run_dir.exists():
                import shutil
                shutil.rmtree(self.run_dir)
            else:
                self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"save is {self.save}, saving experiment to {self.run_dir}")

        # Always create these directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.run_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Additional directories only created when saving is enabled
        if self.save:
            self.tensorboard_dir = self.run_dir / "tensorboard"
            self.tensorboard_dir.mkdir(exist_ok=True)

            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        else:
            self.writer = None
    
    def save_checkpoint_data(self, checkpoint_data):
        """Save checkpoint with custom data."""
        if not self.save:
            return
        
        epoch = checkpoint_data['epoch']
        loss = checkpoint_data.get('train_loss', 0.0)
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        
        self.best_losses.append((loss, checkpoint_path))
        self.best_losses.sort(key=lambda x: x[0])
        
        if len(self.best_losses) > self.keep_best_n:
            _, worst_checkpoint_path = self.best_losses.pop()
            if worst_checkpoint_path.exists():
                worst_checkpoint_path.unlink()
        
        latest_path = self.run_dir / "latest_checkpoint.txt"
        with open(latest_path, 'w') as f:
            f.write(str(checkpoint_path))


    def save_checkpoint(self, epoch, optimizer_i, optimizer_a, loss):
        """Save checkpoint and maintain only the best N checkpoints"""
        if not self.save:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_i_state_dict': self.model_i.state_dict(),
            'model_a_state_dict': self.model_a.state_dict(),
            'optimizer_i_state_dict': optimizer_i.state_dict(),
            'optimizer_a_state_dict': optimizer_a.state_dict(),
            'loss': loss,
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.best_losses.append((loss, checkpoint_path))
        self.best_losses.sort(key=lambda x: x[0])
        
        if len(self.best_losses) > self.keep_best_n:
            _, worst_checkpoint_path = self.best_losses.pop()
            if worst_checkpoint_path.exists():
                worst_checkpoint_path.unlink()
        
        latest_path = self.run_dir / "latest_checkpoint.txt"
        with open(latest_path, 'w') as f:
            f.write(str(checkpoint_path))
    
    def save_final_models(self):
        """Save the final trained models"""
        #if not self.save:
        #    return

        torch.save(self.model_i.state_dict(), self.models_dir / 'model_i_final.pth')
        torch.save(self.model_a.state_dict(), self.models_dir / 'model_a_final.pth')
        
        if self.checkpoint_dir.exists():
            for checkpoint in self.checkpoint_dir.glob("*.pt"):
                checkpoint.unlink()
            self.checkpoint_dir.rmdir()
    
    def save_config(self, config):
        """Save training configuration to JSON"""

        # Check if loss_function is a callable (function) or a string

        # Add model configurations if available
        if hasattr(self.model_i, 'config'):
            config['model_i_config'] = self.model_i.config
        if hasattr(self.model_a, 'config'):
            config['model_a_config'] = self.model_a.config

        config['dataset_name'] = self.dataset_name
    
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def load_checkpoint(self, checkpoint_path):
        """Load models from checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model_i.load_state_dict(checkpoint['model_i_state_dict'])
        self.model_a.load_state_dict(checkpoint['model_a_state_dict'])
        return checkpoint
    
    def train_epoch(self, data_loader, optimizer_i, optimizer_a, epoch):
        """Train for one epoch"""

        self.model_i.train()
        self.model_a.train()

        total_loss = 0
        i = 0
        print(f'Training on dataset: {self.dataset_name}')
        
        for batch_idx, batch in enumerate(data_loader):
            L0, amf, fwpos_values, t_int = batch

            L0 = L0.squeeze(0)
            fwpos_values = fwpos_values.squeeze(0)

            t_int = t_int.transpose(0, 1)
            amf = amf.transpose(0, 1)

            # if amf std is nan skip this batch, as the training will fail
            if torch.isnan(amf.std()).any() or torch.isinf(amf.std()).any():
                print(f"Skipping batch {batch_idx} due to NaN or inf in amf std")
                continue


            L1_i = self.model_i(L0, fwpos_values, t_int)
            L1_i = L1_i.to(torch.float32)

            L1_a = self.model_a(L1_i, amf)

            with torch.no_grad():
                latent_space = self.model_a.encoder(L1_i)

                # Pass all loss function parameters correctly
            current_loss = combined_loss(L1_a, L1_i, latent_space, amf, mse_loss_w=self.mse_loss_w, corr_loss_w=self.corr_loss_w, sam_loss_w=self.sam_loss_w, cos_loss_w = self.cos_loss_w)
            with torch.no_grad():
                residual = L1_a - L1_i
                mean_abs_residual = torch.mean(torch.abs(residual))
            
            optimizer_a.zero_grad()
            optimizer_i.zero_grad()
            current_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model_i.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model_a.parameters(), 1.0)
            
            optimizer_a.step()
            optimizer_i.step()
            
            total_loss += current_loss.item()
            i += 1
            
            if self.writer:
                global_step = epoch * len(data_loader) + batch_idx
                self.writer.add_scalar("Train/Loss", current_loss.item(), global_step)
                self.writer.add_scalar("Train/MeanAbsoluteResidual", mean_abs_residual.item(), global_step)

                for dim in range(latent_space.shape[1]):
                    self.writer.add_scalar(f"LatentSpace/Dim_{dim}", latent_space[:, dim].mean().item(), global_step)
                
                # Log parameter and gradient histograms every 100 batches
                if batch_idx % 100 == 0:
                    for name, param in self.model_i.named_parameters():
                        self.writer.add_histogram(f"InstrumentModel/{name}", param, global_step)
                        if param.grad is not None:
                            self.writer.add_histogram(f"InstrumentModel/{name}.grad", param.grad, global_step)
                    
                    for name, param in self.model_a.named_parameters():
                        self.writer.add_histogram(f"AtmosphericModel/{name}", param, global_step)
                        if param.grad is not None:
                            self.writer.add_histogram(f"AtmosphericModel/{name}.grad", param.grad, global_step)
        
        
            if i % 10 == 0:
                print(f'Batch {i} Loss: {current_loss.item()}')

            
        return total_loss / len(data_loader)


    def close(self):
        """Close the TensorBoard writer"""
        if self.writer:
            self.writer.close()



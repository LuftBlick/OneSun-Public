# src/evaluation/model_evaluation.py

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Optional, Dict, List
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.loss_functions import combined_loss


class ModelEvaluator:
    """
    Class for evaluating trained models using the experiment directory structure
    created by ModelTrainer
    """

    def __init__(self, trainer=None, experiment_dir: Optional[Path] = None):
        """
        Initialize evaluator either with a trainer instance or experiment directory

        Args:
            trainer: Optional ModelTrainer instance
            experiment_dir: Optional Path to experiment directory
        """
        save = getattr(trainer, 'save', False) if trainer is not None else False

        if trainer is not None:
            self.run_dir = trainer.run_dir
            self.models_dir = trainer.models_dir
            self.checkpoint_dir = trainer.checkpoint_dir
        elif experiment_dir is not None:
            self.run_dir = Path(experiment_dir)
            self.models_dir = self.run_dir / "models"
            self.checkpoint_dir = self.run_dir / "checkpoints"
        elif trainer is None and experiment_dir is None:
            raise ValueError("Must provide either trainer or experiment_dir")

        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from the experiment directory"""
        config_path = self.run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found in {self.run_dir}")

        with open(config_path, 'r') as f:
            return json.load(f)

    def load_models(self, model_i, model_a, checkpoint_epoch: Optional[int] = None) -> int:
        """
        Load models from checkpoint or final saved models

        Args:
            model_i: Initialized InstrumentModel instance
            model_a: Initialized AtmosphericModel instance
            checkpoint_epoch: Optional specific epoch to load

        Returns:
            epoch number (-1 for final model)
        """
        if checkpoint_epoch is not None:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{checkpoint_epoch}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint for epoch {checkpoint_epoch} not found")

            checkpoint = torch.load(checkpoint_path)
            model_i.load_state_dict(checkpoint['model_i_state_dict'])
            model_a.load_state_dict(checkpoint['model_a_state_dict'])
            return checkpoint['epoch']
        else:
            # Load final models
            model_i_path = self.models_dir / "model_i_final.pth"
            model_a_path = self.models_dir / "model_a_final.pth"

            if not model_i_path.exists() or not model_a_path.exists():
                raise FileNotFoundError("Final model files not found")

            model_i.load_state_dict(torch.load(model_i_path))
            model_a.load_state_dict(torch.load(model_a_path))
            return -1

    def evaluate_models(self, model_i, model_a, dataloader, device='cpu', mse_loss_w=0.1, corr_loss_w=0.1,
                        sam_loss_w=0.1, cos_loss_w=0.1) -> (list, float):
        """
        Evaluate models on a dataset and return predictions.

        Args:
            model_i: Trained InstrumentModel.
            model_a: Trained AtmosphericModel.
            dataloader: DataLoader instance.
            device: Device for evaluation.

        Returns:
            results: List of dictionaries with predictions and other outputs.
            avg_loss: Average loss over all batches (or None if loss_fn is not provided).
        """
        model_i.eval()
        model_a.eval()
        results = []
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                L0, amf, fwpos_values, t_int = batch

                # Adjust dimensions as needed (similar to your current code)
                L0 = L0.squeeze(0)
                fwpos_values = fwpos_values.squeeze(0)
                t_int = t_int.transpose(0, 1)
                amf = amf.transpose(0, 1)

                # Move to the specified device
                L0 = L0.to(device)
                amf = amf.to(device)
                fwpos_values = fwpos_values.to(device)
                t_int = t_int.to(device)

                # Forward pass
                L1_i = model_i(L0, fwpos_values, t_int)
                L1_i = L1_i.to(torch.float32)
                L1_a = model_a(L1_i, amf)

                batch_loss = None
                latent_space = model_a.encoder(L1_i)
                # Compute the combined loss using L1_a, L1_i, latent_space (as totcol), and amf.
                loss = combined_loss(L1_a, L1_i, latent_space, amf, mse_loss_w, corr_loss_w, sam_loss_w, cos_loss_w)
                batch_loss = loss.item()
                total_loss += batch_loss
                count += 1

                # Store results (including loss if computed)
                results.append({
                    'L0': L0.cpu(),
                    'L1_i': L1_i.cpu(),
                    'L1_a': L1_a.cpu(),
                    'amf': amf.cpu(),
                    'fwpos_values': fwpos_values.cpu(),
                    't_int': t_int.cpu(),
                    'loss': batch_loss
                })

        avg_loss = total_loss / count if count > 0 else None
        return results, avg_loss

    def plot_predictions(self, L0, L1_i, L1_a, wavelengths=None, title=None) -> go.Figure:
        """
        Plot the original L0 spectrum and model predictions using Plotly
        """
        # Convert tensors to numpy if needed
        if torch.is_tensor(L0):
            L0 = L0.detach().cpu().numpy()
        if torch.is_tensor(L1_i):
            L1_i = L1_i.detach().cpu().numpy()
        if torch.is_tensor(L1_a):
            L1_a = L1_a.detach().cpu().numpy()

        x = wavelengths if wavelengths is not None else np.arange(len(L0))

        # Define custom colors
        color_L0 = 'grey'  # Greyish for original spectrum
        color_L1_i = 'lightblue'  # Light bluish for instrument model
        color_L1_a = 'blue'  # Darker bluish for atmospheric model
        color_residual = 'grey'  # Same greyish for residuals

        # Create subplots: 2 rows, shared x-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Spectrum", "Residuals")
        )

        # Top: spectra
        fig.add_trace(
            go.Scatter(
                x=x, y=L0, name='L0 (Original)', mode='lines', line=dict(color=color_L0)
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=L1_i, name='L1_i (Instrument Model)', mode='lines', line=dict(color=color_L1_i)
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=L1_a, name='L1_a (Atmospheric Model)', mode='lines', line=dict(color=color_L1_a)
            ), row=1, col=1
        )

        # Bottom: residuals
        residual = L1_a - L1_i
        fig.add_trace(
            go.Scatter(
                x=x, y=residual, name='Residual (L1_a - L1_i)', mode='lines', line=dict(color=color_residual)
            ), row=2, col=1
        )

        # Axis labels
        fig.update_yaxes(title_text='Intensity', row=1, col=1)
        fig.update_yaxes(title_text='Residual', row=2, col=1)
        fig.update_xaxes(title_text='Wavelength' if wavelengths is not None else 'Pixel', row=2, col=1)

        # Layout
        fig.update_layout(
            title_text=title,
            height=600,
            width=800,
            showlegend=True
        )
        return fig

    def plot_all_spectra_heatmap(self, results, var='L0', wavelengths=None, title=None):
        # 1) build S×P matrix
        mats = []
        for batch in results:
            arr = batch[var].cpu().numpy()
            # make sure it’s 2D: (n_spectra, n_pixels)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            mats.append(arr)
        mat = np.vstack(mats)

        # 2) render
        x = wavelengths if wavelengths is not None else np.arange(mat.shape[1])
        fig = px.imshow(
            mat,
            x=x,
            y=np.arange(mat.shape[0]),
            labels={'x': 'Pixel/Wavelength', 'y': 'Spectrum Index', 'color': var},
            aspect='auto',
            title=title or f"Heatmap of all {var} Spectra"
        )
        return fig

    def plot_small_multiples(self, results, var='L1_a', ncols=5, wavelengths=None, title=None):
        # gather flat list of 1D arrays
        mats = []
        for batch in results:
            arr = batch[var].cpu().numpy()
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            mats.extend(arr)
        n = len(mats)
        nrows = int(np.ceil(n / ncols))

        fig = make_subplots(
            rows=nrows, cols=ncols,
            shared_xaxes=True, shared_yaxes=True,
            subplot_titles=[f"S{i}" for i in range(n)]
        )
        x = wavelengths if wavelengths is not None else np.arange(mats[0].shape[0])
        for idx, spec in enumerate(mats):
            r, c = divmod(idx, ncols)
            fig.add_trace(
                go.Scatter(x=x, y=spec, mode='lines', line=dict(width=1)),
                row=r + 1, col=c + 1
            )
            fig.update_xaxes(visible=False, row=r + 1, col=c + 1)
            fig.update_yaxes(visible=False, row=r + 1, col=c + 1)

        fig.update_layout(
            height=200 * nrows,
            width=200 * ncols,
            title=title or f"Small Multiples of {var} Spectra",
            showlegend=False
        )
        return fig

    def plot_pixel_overview(self, results, pixel_indices=[500, 1000, 1500], var='L0', ncols=6, title=None) -> go.Figure:
        """
        Plot time-series of specified pixel indices across each batch/day.
        Each subplot is one day; within each, lines show pixel intensity across spectra.
        """
        n_days = len(results)
        nrows = int(np.ceil(n_days / ncols))
        fig = make_subplots(
            rows=nrows, cols=ncols,
            shared_xaxes=True, shared_yaxes=False,
            subplot_titles=[f"Day {i}" for i in range(n_days)]
        )
        colors = ['grey', 'lightblue', 'blue']
        for day_idx, batch in enumerate(results):
            arr = batch[var]
            if torch.is_tensor(arr): arr = arr.detach().cpu().numpy()
            if arr.ndim == 1: arr = arr[np.newaxis, :]
            x = np.arange(arr.shape[0])
            row = day_idx // ncols + 1
            col = day_idx % ncols + 1
            for pi_idx, pix in enumerate(pixel_indices):
                y = arr[:, pix]
                show_leg = (day_idx == 0)
                fig.add_trace(
                    go.Scatter(
                        x=x, y=y,
                        mode='lines',
                        name=f"Pixel {pix}",
                        line=dict(color=colors[pi_idx]),
                        showlegend=show_leg
                    ),
                    row=row, col=col
                )
            # fig.update_xaxes(title_text='Spectrum Index', row=row, col=col)
            # fig.update_yaxes(title_text=var, row=row, col=col)
        fig.update_layout(
            height=200 * nrows,
            width=200 * ncols,
            title_text=title or f"Pixel Overview: {var}",
            showlegend=True
        )
        return fig

    def plot_predictions_mpl(self, L0, L1_i, L1_a, wavelengths=None,
                             title=None, figsize=(12, 8)) -> plt.Figure:
        """
        Plot the original L0 spectrum and model predictions

        Args:
            L0: Original L0 spectrum (torch.Tensor or numpy array)
            L1_i: Instrument model prediction (torch.Tensor or numpy array)
            L1_a: Atmospheric model prediction (torch.Tensor or numpy array)
            wavelengths: Optional wavelength array for x-axis
            title: Optional plot title
            figsize: Figure size tuple

        Returns:
            matplotlib Figure object
        """
        # Convert tensors to numpy if needed
        if torch.is_tensor(L0):
            L0 = L0.detach().cpu().numpy()
        if torch.is_tensor(L1_i):
            L1_i = L1_i.detach().cpu().numpy()
        if torch.is_tensor(L1_a):
            L1_a = L1_a.detach().cpu().numpy()

        x_l0 = wavelengths if wavelengths is not None else np.arange(len(L0))
        x_l1 = wavelengths if wavelengths is not None else np.arange(len(L1_i))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot spectra
        ax1.plot(x_l0, L0, label='L0 (Original)', alpha=0.7)
        ax1.plot(x_l1, L1_i, label='L1_i (Instrument Model)', alpha=0.7)
        ax1.plot(x_l1, L1_a, label='L1_a (Atmospheric Model)', alpha=0.7)
        ax1.set_ylabel('Intensity')
        ax1.legend()
        ax1.grid(True)

        # Plot residuals
        residual_a = L1_a - L1_i
        ax2.plot(x_l1, residual_a, label='L1_a - L1_i', alpha=0.7)
        ax2.set_ylabel('Residual')
        ax2.set_xlabel('Wavelength (nm)' if wavelengths is not None else 'Pixel')
        ax2.legend()
        ax2.grid(True)

        if title:
            fig.suptitle(title)
        plt.tight_layout()
        return fig

    def plot_latent_space(self, model_a, L1_i, amf) -> plt.Figure:
        """
        Plot the latent space variables

        Args:
            model_a: Trained AtmosphericModel
            L1_i: Instrument model output
            amf: Air mass factors

        Returns:
            matplotlib Figure object
        """
        model_a.eval()
        with torch.no_grad():
            latent_space = model_a.encoder(L1_i)

        latent_space = latent_space.detach().cpu().numpy()
        amf = amf.cpu().numpy()

        fig, axes = plt.subplots(latent_space.shape[1], 1,
                                 figsize=(10, 3 * latent_space.shape[1]))
        if latent_space.shape[1] == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.scatter(amf, latent_space[:, i], alpha=0.5)
            ax.set_xlabel('Air Mass Factor')
            ax.set_ylabel(f'Latent Variable {i + 1}')
            ax.grid(True)

        plt.tight_layout()
        return fig

    def plot_loss_curves(self, train_losses, valid_losses, title="Training vs. Validation Loss", figsize=(10, 6)):
        """
        Plot and compare training and validation loss curves.

        Args:
            train_losses (list or array): List of training loss values per epoch.
            valid_losses (list or array): List of validation loss values per epoch.
            title (str): Title for the plot.
            figsize (tuple): Figure size.

        Returns:
            matplotlib.figure.Figure: The generated loss curves plot.
        """
        epochs = range(1, len(train_losses) + 1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(epochs, train_losses, marker='o', label="Training Loss")
        ax.plot(epochs, valid_losses, marker='o', label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        return fig




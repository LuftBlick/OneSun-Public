import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


"""
Configuration settings for the training pipeline.
"""
from utils.loss_functions import combined_loss, spectral_fidelity_loss

# Dataset configuration
def get_default_dataset_config():
    return {
        'l0_dataset_name': 'Pandora117s1_Rome-SAP_20240701_20241231.pkl',
        'train_date_range': ('2024-07-01', '2024-07-10'),
        'val_date_range': ('2024-07-11', '2024-07-15'),
        'dataset_name': 'Pandora117s1_Rome_SAP_test1',  # Human-readable name
        'latitude': 41.9028,  # Rome coordinates
        'longitude': 12.4964,
        'altitude': 21,  # meters
    }

# Training configuration
def get_default_training_config():
    return {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 10,
        'batch_size': 1,
        'dataset_name': 'Pandora117s1_Rome_SAP_test1',  # 
        'save_dir': '../experiment_runs/',
        'experiment_name': 'default_experiment',
        'instrument_model': 'Instrument Model',
        'atmospheric_model': 'Atmospheric Model',
        'save': True,
        'loss_function': combined_loss
    }


"""
Model initialization functions.
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from models.atmospheric_model import AtmosphericModel
from models.instrument_model import InstrumentModel
from utils.atmos import get_tau_rayleigh

def load_constants(data_dir, gas_ods, solar_conv):
    """Load cross-sections, solar spectrum, and calculate Rayleigh scattering."""
    #base_dir = Path(__file__).resolve().parent.parent 
    #data_dir = base_dir / "data" / "constants"
    print("loading constants from", data_dir)
    xsec_gas = pd.read_pickle(f'{data_dir}/{gas_ods}')
    le_solar = pd.read_pickle(f'{data_dir}/{solar_conv}')

    wl = xsec_gas.iloc[:, 0]
    tau_rayleigh = get_tau_rayleigh(wl, 0)
    
    # Cloud coefficients
    cloud_df = pd.DataFrame({
        'ones': np.ones(len(wl)),
        'wl': list(wl),
        'wl2': list(wl**2)
    }, index=wl.index)
    
    # Scale cloud coefficients
    min_val = np.min(cloud_df.values)
    max_val = np.max(cloud_df.values)
    cloud_df = cloud_df.sub(min_val).div(max_val - min_val)
    
    return xsec_gas, le_solar, tau_rayleigh, cloud_df, wl

def init_models(data_sample, atmospheric_model="AtmosphericModel", instrument_model="InstrumentModel", gas_ods=" gas_ods_conv05.pkl", solar_conv="solar_conv05.pkl"):
    """
    Initialize instrument and atmospheric models based on input data dimensions.
    
    Args:
        data_sample: A sample from the dataset to infer dimensions
        atmospheric_model: Class name of the atmospheric model
        instrument_model: Class name of the instrument model
        
    Returns:
        model_i: Initialized InstrumentModel
        model_a: Initialized AtmosphericModel
    """
    # Unpack sample to infer dimensions
    spectral_tensor, amf_tensor, fwpos_tensor, t_int_tensor = data_sample
    
    # Infer dimensions
    input_dim = spectral_tensor.shape[1]
    fwpos_dim = fwpos_tensor.shape[1]
    
    # Load constants
    print("Loading constants...")
    const_dir = os.environ.get("ONESUN_CONST_DIR", '../../data/constants')
    print("Loading constants from", const_dir)
    xsec_gas, le_solar, tau_rayleigh, cloud_df, _ = load_constants(data_dir=const_dir, gas_ods=gas_ods, solar_conv=solar_conv)
    # in ipynbs:
    #xsec_gas, le_solar, tau_rayleigh, cloud_df, _ = load_constants(data_dir='../data/constants')

    # Initialize models
    l1_dimensions = 2000
    latent_dim = len(xsec_gas.columns) - 1


    model_i = instrument_model_chooser(instrument_model, input_dim, fwpos_dim, l1_dimensions)

    model_a = atmospheric_model_chooser(atmospheric_model, l1_dimensions, latent_dim, l1_dimensions, xsec_gas.iloc[:, 1:], le_solar.iloc[:, 1:], tau_rayleigh)

    # Store configuration for future initialization
    model_i.config = {
        'input_dim': input_dim,
        'fwpos_dim': fwpos_dim,
        'output_dim': l1_dimensions
    }
    
    model_a.config = {
        'input_dim': l1_dimensions,
        'latent_dim': latent_dim,
        'output_dim': l1_dimensions,
        # We can't easily serialize these, so we'll store their shapes only
        'xsec_gas_shape': xsec_gas.shape,
        'le_solar_shape': le_solar.shape,
        'tau_rayleigh_shape': tau_rayleigh.shape if hasattr(tau_rayleigh, 'shape') else None
    }
    
    return model_i, model_a

def init_optimizers(model_i, model_a, config):
    """Initialize optimizers for both models."""
    optimizer_i = torch.optim.Adam(
        model_i.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    print(f"Instrument model parameters: {len(list(model_i.parameters()))}")
    params_a = list(model_a.parameters()) if model_a is not None else []
    
    if not params_a:
        print("Warning: Atmospheric model has no parameters to optimize.")
        optimizer_a = None
    else:
        optimizer_a = torch.optim.Adam(
            params_a,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    
    return optimizer_i, optimizer_a


import importlib

def atmospheric_model_chooser(atmospheric_model_module_name: str, input_dim: int, latent_dim: int, output_dim: int, xsec_gas, le_solar, tau_rayleigh):
    try:
        class_name = "AtmosphericModel"

        module_path = f"models.{atmospheric_model_module_name}"
        model_module = importlib.import_module(module_path)

        # Get the class from the module
        ModelClass = getattr(model_module, class_name)
        print(f"Instantiating model class: {ModelClass.__name__} from module: {module_path}")

        # Instantiate the model
        # Ensure the model's __init__ signature matches these arguments
        model_a = ModelClass(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            xsec_list=xsec_gas,
            le_list=le_solar,
            tau_rayleigh=tau_rayleigh
        )
        return model_a

    except ImportError:
        raise ValueError(
            f"Could not import atmospheric model module: {module_path}. Ensure the file exists and 'src' is in PYTHONPATH.")
    except AttributeError:
        raise ValueError(f"Class '{class_name}' not found in module '{module_path}'. Check class naming conventions.")
    except Exception as e:
        raise ValueError(f"Error instantiating atmospheric model '{class_name}' from '{module_path}': {e}")

def instrument_model_chooser(instrument_model_module_name: str, input_dim: int, fwpos_dim: int, l1_dimensions: int):
    try:
        class_name = "InstrumentModel"
        # model path?
        print(f"Loading instrument model '{class_name}' from module '{instrument_model_module_name}'")
        module_path = f"models.{instrument_model_module_name}"
        print(f"Module path: {module_path}")
        model_module = importlib.import_module(module_path)

        # Get the class from the module
        ModelClass = getattr(model_module, class_name)
        print(f"Instantiating model class: {ModelClass.__name__} from module: {module_path}")

        # Instantiate the model
        # Ensure the model's __init__ signature matches these arguments
        model_i = ModelClass(
            input_dim=input_dim,
            fwpos_dim=fwpos_dim,
            output_dim=l1_dimensions
        )
        return model_i

    except ImportError:
        raise ValueError(
            f"Could not import instrument model module: {module_path}. Ensure the file exists and 'src' is in PYTHONPATH.")
    except AttributeError:
        raise ValueError(f"Class '{class_name}' not found in module '{module_path}'. Check class naming conventions.")
    except Exception as e:
        raise ValueError(f"Error instantiating instrument model '{class_name}' from '{module_path}': {e}")
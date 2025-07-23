import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
#from src.utils.atmos import calculate_sza_am  # the function to compute solar zenith angle and air mass
import os
import sys
from pathlib import Path


#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#project_root = Path(__file__).resolve().parent.parent.parent#.parent
#src_path = project_root / "src"
#print(f"DEBUG: Current src path: {src_path}")
#sys.path.insert(0, str(src_path))


from utils.atmos import calculate_sza_am  # the function to compute solar zenith angle and air mass


class DailyDataset(Dataset):
    def __init__(self, spectral_data: pd.DataFrame, amf: pd.Series, fwpos_df: pd.DataFrame, t_int: pd.Series):
        """
        Please note, the arguments must be DataFrames or Series with an index of datetime objects.
        """
        self.spectral_data = spectral_data
        self.amf = amf
        self.fwpos_df = fwpos_df
        self.t_int = t_int
        self.days = spectral_data.index.normalize().unique()

    def __len__(self):
        return len(self.days)

    def __getitem__(self, idx):
        day = self.days[idx]
        day_data = self.spectral_data.loc[day.strftime('%Y-%m-%d')]
        day_amf = self.amf.loc[day.strftime('%Y-%m-%d')]
        day_fwpos = self.fwpos_df.loc[day.strftime('%Y-%m-%d')]
        day_t_int = self.t_int.loc[day.strftime('%Y-%m-%d')]

        return (torch.tensor(day_data.values, dtype=torch.float32),
                torch.tensor(day_amf.values, dtype=torch.float32),
                torch.tensor(day_fwpos.values, dtype=torch.float32),
                torch.tensor(day_t_int.values, dtype=torch.float32))





def load_and_split_dataset(pickle_path: str,
                           train_date_range: tuple,
                           val_date_range: tuple,
                           latitude: float,
                           longitude: float,
                           altitude: float = 0) -> (DailyDataset, DailyDataset):
    """
    Load the full dataset from a pickle file, split it into training and validation subsets
    based on date ranges, process each subset, and return DailyDataset instances for each.
        
    """
    # Load the full dataset from the pickle file
    print(f'loading df from {pickle_path}') 
    print(f'of size {os.path.getsize(pickle_path)} bytes')

    df_full = pd.read_pickle(pickle_path)
    print('loaded df')
    
    # Ensure the index is a datetime index
    if not isinstance(df_full.index, pd.DatetimeIndex):
        df_full.index = pd.to_datetime(df_full.index)
    
    # Subset the dataframe based on the provided date ranges
    df_train = df_full.loc[train_date_range[0]:train_date_range[1]]
    df_val = df_full.loc[val_date_range[0]:val_date_range[1]]
    
    def process_df(df_subset: pd.DataFrame):
        # Compute air mass using the SZA calculation; here we use am_gas
        amf = calculate_sza_am(df_subset, latitude, longitude, altitude)['am_gas']
        
        # Integration time
        t_int = df_subset['integration_time']
        
        # Encode filterwheel positions as one-hot vectors
        fwpos1 = df_subset['filterwheel_pos1'].astype(int)
        fwpos2 = df_subset['filterwheel_pos2'].astype(int)
        fwpos1_encoded = pd.get_dummies(fwpos1, prefix='fwpos1').astype(int)
        fwpos2_encoded = pd.get_dummies(fwpos2, prefix='fwpos2').astype(int)
        fwpos_df = pd.concat([fwpos1_encoded, fwpos2_encoded], axis=1)
        
        # Process spectral data:
        # Convert the 'data' column (which is a list of spectra) into a DataFrame
        spectral_data = df_subset['data']
        spectral_df = pd.DataFrame(spectral_data.values.tolist(), index=spectral_data.index)
        
        # Scale the spectral data to [0, 1]
        min_val = spectral_df.values.min()
        max_val = spectral_df.values.max()
        spectral_df_scaled = spectral_df.sub(min_val).div(max_val - min_val)
        
        return spectral_df_scaled, amf, fwpos_df, t_int
    
    # Process training and validation subsets
    spec_train, amf_train, fwpos_train, t_int_train = process_df(df_train)
    spec_val, amf_val, fwpos_val, t_int_val = process_df(df_val)
    print('processed df')
    
    # Create DailyDataset instances
    train_dataset = DailyDataset(spec_train, amf_train, fwpos_train, t_int_train)
    val_dataset = DailyDataset(spec_val, amf_val, fwpos_val, t_int_val)
    print('created datasets')
    
    return train_dataset, val_dataset


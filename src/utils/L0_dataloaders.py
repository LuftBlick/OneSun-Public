import pandas as pd
import numpy as np
import requests
import bz2
import re
import json
import os
import sys
from urllib.parse import urlparse


def download_and_uncompress_bz_to_string(url: str) -> str:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        bz_content = response.content
    decompressed_content = bz2.decompress(bz_content)
    return decompressed_content.decode('latin-1')


def download_txt_file_to_string(url: str) -> str:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        content = response.text
    return content


def read_local_bz_to_string(filepath: str) -> str:
    """Read and uncompress a local .bz2 file"""
    with bz2.open(filepath, 'rt', encoding='latin-1') as file:
        return file.read()


def read_local_txt_to_string(filepath: str) -> str:
    """Read a local text file"""
    with open(filepath, 'r', encoding='latin-1') as file:
        return file.read()


def find_section_boundaries(content: [str]) -> tuple:
    """Find the boundaries for the three sections"""
    separator_indices = [i for i, line in enumerate(content) 
                        if line.startswith('------------------')]
    if len(separator_indices) < 2:
        raise ValueError("File format error: Could not find section separators")
    return separator_indices[0], separator_indices[1]


def parse_metadata(file_content: str) -> dict:
    """Extract metadata section"""
    content_lines = file_content.splitlines()
    meta_end_idx = find_section_boundaries(content_lines)[0]
    
    metadata = {}
    for line in content_lines[:meta_end_idx]:
        if ':' in line:
            try:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
            except ValueError:
                continue
    return metadata

def parse_column_definitions(file_content: str) -> dict:
    """Extract column definitions and their positions"""
    content_lines = file_content.splitlines()
    meta_end_idx, coldef_end_idx = find_section_boundaries(content_lines)
    
    column_positions = {}
    spectrum_range = None
    
    for line in content_lines[meta_end_idx + 1:coldef_end_idx]:
        if ': ' in line and line.startswith('Column'):
            col_def, description = line.split(': ', 1)
            col_parts = col_def.strip().split()
            
            # Handle column ranges (for spectrum data)
            if len(col_parts) == 2 and '-' in col_parts[1]:
                if "Mean over all cycles" in description:
                    #print(col_parts)
                    start, end = map(int, col_parts[1].split('-'))
                    #print(start, end)
                    spectrum_range = (start - 1, end - 1)  # Convert to 0-based indexing
            else:
                # Single column
                try:
                    col_num = int(col_parts[1])
                    column_positions[description.strip()] = col_num - 1  # Convert to 0-based indexing
                except (IndexError, ValueError):
                    continue
    
    if spectrum_range:
        column_positions['spectrum_range'] = spectrum_range
        
    return column_positions

def get_data_lines(file_content: str) -> [str]:
    """Get all data lines after the last separator"""
    content_lines = file_content.splitlines()
    _, coldef_end_idx = find_section_boundaries(content_lines)
    
    data_lines = []
    for line in content_lines[coldef_end_idx + 1:]:
        # All lines that start with routine code (2 letters) or '**' and have a timestamp
        if (re.match(r"[A-Z]{2}\s\d{8}T\d{6}", line) or 
            (line.startswith('** ') and 'T' in line)):
            if not line.endswith('ERROR') and not line.endswith('INFO'):
                data_lines.append(line)
    return data_lines


def df_from_string(file_content: str, columns_to_read: list = None, column_mapping_file: str = '../src/utils/L0_column_mappings.json') -> tuple:
    """
    Create DataFrame from Pandora L0 file content.
    Required columns: time, scale_factor
    Optional columns: detector_temp, spectrometer_temp, etc.
    
    Args:
        file_content (str): Content of the L0 file
        columns_to_read (list): List of columns to read
        column_mapping_file (str): Path to JSON file with column mappings
        
    Returns:
        tuple: (main_df, skipped_df)
    """
    import warnings
    
    # Load column mappings
    with open(column_mapping_file, 'r') as f:
        column_mappings = json.load(f)
    
    # Get file structure
    try:
        column_positions = parse_column_definitions(file_content)
    except ValueError as e:
        raise RuntimeError(f"Invalid file structure: {str(e)}")
    
    # Required columns that must exist
    REQUIRED_COLUMNS = {'time', 'scale_factor'}
    
    # Initialize data structures
    data = []
    spectrum_data = []
    skipped_lines = []
    
    # Prepare columns to read
    if columns_to_read is None:
        columns_to_read = list(column_mappings.keys())
    if 'scale_factor' not in columns_to_read:
        columns_to_read.append('scale_factor')
    
    # Map column names to positions
    cols_to_read = []
    final_column_names = []
    
    for short_name in columns_to_read:
        if short_name not in column_mappings:
            warnings.warn(f"Unknown column name: {short_name}")
            continue
            
        long_name = column_mappings[short_name]
        pos = None
        for desc, p in column_positions.items():
            if desc == long_name:
                pos = p
                break
                
        if pos is None:
            if short_name in REQUIRED_COLUMNS:
                raise ValueError(f"Required column not found in file: {long_name}")
            else:
                warnings.warn(f"Optional column not found in file: {long_name}")
                continue
                
        cols_to_read.append(pos)
        final_column_names.append(short_name)
    
    # Verify we have all required columns
    found_columns = set(final_column_names)
    missing_required = REQUIRED_COLUMNS - found_columns
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    
    # Get spectrum range
    spectrum_range = column_positions.get('spectrum_range')
    if not spectrum_range:
        raise ValueError("Spectrum range not found in file")
    
    # Process data lines
    content_lines = file_content.splitlines()
    _, coldef_end_idx = find_section_boundaries(content_lines)
    
    for line in content_lines[coldef_end_idx + 1:]:
        line = line.strip()
        if not line:
            continue
            
        # Check if line has timestamp format
        if not re.search(r"\d{8}T\d{6}", line):
            continue
            
        fields = line.split()
        
        # Skip informational lines but track them
        if any(line.endswith(suffix) for suffix in ['ERROR', 'INFO']):
            skipped_lines.append({
                'line': line,
                'reason': 'Status message',
                'timestamp': fields[1]
            })
            continue
            
        if '#' in line:
            skipped_lines.append({
                'line': line,
                'reason': 'Comment line',
                'timestamp': fields[1]
            })
            continue
        
        try:
            # Get all required columns
            if len(fields) <= max(cols_to_read):
                raise ValueError("Line has fewer fields than required")
            
            row = [fields[i] for i in cols_to_read]
            
            # Validate scale factor (using its position from cols_to_read)
            scale_factor_idx = final_column_names.index('scale_factor')
            try:
                scale = float(row[scale_factor_idx])
                if scale <= 0:
                    raise ValueError("Scale factor must be positive")
            except (ValueError, TypeError):
                raise ValueError("Invalid scale factor")
            
            # Process spectrum data
            if len(fields) <= spectrum_range[1]:
                raise ValueError("Missing spectrum data")
            
            start, end = spectrum_range
            try:
                spectrum = np.array([float(x) for x in fields[start:end + 1]])
                spectrum = spectrum / scale
                spectrum_data.append(spectrum)
                data.append(row)
            except (ValueError, TypeError):
                raise ValueError("Invalid spectrum values")
                
        except ValueError as e:
            skipped_lines.append({
                'line': line,
                'reason': str(e),
                'timestamp': fields[1] if len(fields) > 1 else 'unknown'
            })
            continue
    
    if not data:
        raise ValueError("No valid data lines found in file")
    
    # Create main DataFrame
    main_df = pd.DataFrame(data, columns=final_column_names)
    
    # Convert numeric columns
    numeric_cols = ['integration_time', 'cycles', 'detector_temp', 'spectrometer_temp', 'scale_factor']
    for col in numeric_cols:
        if col in main_df.columns:
            main_df[col] = pd.to_numeric(main_df[col], errors='coerce')
            if col == 'scale_factor':
                invalid_sf = main_df['scale_factor'].isna() | (main_df['scale_factor'] <= 0)
                if invalid_sf.any():
                    raise ValueError("Found invalid scale factors after numeric conversion")
    
    # Convert time and set index
    if 'time' in main_df.columns:
        main_df['time'] = pd.to_datetime(main_df['time'], format='%Y%m%dT%H%M%S.%fZ')
        main_df.set_index('time', inplace=True)
    
    # Add spectrum data
    main_df['data'] = spectrum_data
    
    # Remove scale_factor if it wasn't originally requested
    #if 'scale_factor' not in columns_to_read:
    main_df = main_df.drop(columns=['scale_factor'])
    
    # Create skipped lines DataFrame
    skipped_df = pd.DataFrame(skipped_lines)
    if not skipped_df.empty:
        skipped_df['timestamp'] = pd.to_datetime(skipped_df['timestamp'], 
                                               format='%Y%m%dT%H%M%S.%fZ', 
                                               errors='coerce')
        skipped_df.set_index('timestamp', inplace=True)
        skipped_df = skipped_df.sort_index()
    
    return main_df, skipped_df




def read_L0file(path, columns_to_read=['time', 'routine_code']):
    """
    Universal wrapper to read L0 files from either URL or local path.
    Supports both .txt and .bz2 formats.
    
    Args:
        path (str): URL or file path to the L0 file
        columns_to_read (list): List of columns to read (default: ['time', 'routine_code'])
        
    Returns:
        tuple: (main_df, skipped_df)
    """
    # Determine if path is URL or local file
    parsed = urlparse(path)
    is_url = bool(parsed.scheme and parsed.netloc)
    
    # Determine file type (.txt or .bz2)
    is_bz2 = path.endswith('.bz2')
    
    print(f"Reading file from {path}")
    # Get file content as string
    if is_url:
        if is_bz2:
            file_string = download_and_uncompress_bz_to_string(path)
        else:
            file_string = download_txt_file_to_string(path)
    else:
        if is_bz2:
            file_string = read_local_bz_to_string(path)
        else:
            file_string = read_local_txt_to_string(path)
    
    # Process file content
    main_df, skipped_df = df_from_string(file_string, columns_to_read)
    
    return main_df, skipped_df



def apply_dark_correction(df: pd.DataFrame, fw_comb_closed: dict = {"fw1": 6, "fw2": 3}) -> pd.DataFrame:
    """
    Apply dark correction to spectral measurements by subtracting the dark measurement 
    (identified by filterwheel positions 6 and 3) from all spectra within the same routine.
    
    Args:
        df: DataFrame with columns for routine_count, repetition_count, filterwheel_pos1, 
            filterwheel_pos2, and data (where data contains the spectral measurements)
    
    Returns:
        DataFrame with dark-corrected spectra
    """
    # Create a copy to avoid modifying the original dataframe
    df_corrected = df.copy()
    
    # Convert string representation of list to actual list if needed
    if isinstance(df_corrected['data'].iloc[0], str):
        df_corrected['data'] = df_corrected['data'].apply(eval)
    
    # Group by routine count
    for routine_num, routine_group in df_corrected.groupby('routine_count'):
        # Find the dark measurement (last repetition with fw_pos 6 and 3)
        dark_measurement = routine_group[
            (routine_group['filterwheel_pos1'] == fw_comb_closed["fw1"]) & 
            (routine_group['filterwheel_pos2'] == fw_comb_closed["fw2"])
        ]
        
        if not dark_measurement.empty:
            # Get the dark spectrum
            dark_spectrum = dark_measurement['data'].iloc[0]
            
            # Get indices for all measurements in this routine
            routine_indices = routine_group.index
            
            # Apply dark correction to all measurements in this routine
            df_corrected.loc[routine_indices, 'data'] = df_corrected.loc[routine_indices, 'data'].apply(
                lambda x: [val - dark for val, dark in zip(x, dark_spectrum)]
            )
    
    return df_corrected


def filter_dark_measurements(df: pd.DataFrame, fw_comb_closed: dict = {"fw1": 6, "fw2": 3}) -> pd.DataFrame:
    """
    Filter out dark measurements (filterwheel positions 6 and 3) from the dataset.
    """
    return df[~((df['filterwheel_pos1'] == fw_comb_closed["fw1"]) & (df['filterwheel_pos2'] == fw_comb_closed["fw2"]))]


# norm by integration time, data/integration_time
def normalize_by_integration_time(df):
    """Normalize spectral data by integration time"""
    
    df_normalized = df.copy()
           
    # Normalize each spectrum by its integration time
    df_normalized['data'] = df_normalized.apply(
        lambda row: [val / row['integration_time'] for val in row['data']], 
        axis=1
    )
    return df_normalized



def cloudscreen_df_ori(df,lim=.02):
    '''
    simple cloud screening based on std of 3 consecutive measurements
    '''
    #index to filter df
    ind = df.rolling(window=3).std()/df.rolling(window=3).max()<lim
    
    return ind

def cloudscreen_df(df, time_window='180s', pixel_idx=1000, lim=0.02):
    """
    Cloud screening based on variability within a time window for a specific pixel

    """
    # Extract the specified pixel from each spectrum
    pixel_series = pd.Series(
        [spectrum[pixel_idx] for spectrum in df['data']],
        index=df.index
    )
    
    # Calculate rolling statistics within time window
    rolling = pixel_series.rolling(time_window, min_periods=3)
    std = rolling.std()
    max_val = rolling.max()
    
    # Calculate ratio and create mask
    ratio = std / max_val
    mask = ratio < lim
    
    return mask



# L2 ##############################

def df_from_L2_string(file_content: str, columns_to_read: list = None, column_mapping_file: str = 'L2_column_mappings.json') -> pd.DataFrame:
    """
    Create DataFrame from Pandora L2 file content.
    
    Args:
        file_content (str): Content of the L2 file
        columns_to_read (list): List of columns to read
        column_mapping_file (str): Path to JSON file with column mappings
        
    Returns:
        pd.DataFrame: DataFrame containing the requested columns
    """
    import warnings
    
    # Load column mappings
    with open(column_mapping_file, 'r') as f:
        column_mappings = json.load(f)
    
    # Get file structure
    try:
        column_positions = parse_column_definitions(file_content)
    except ValueError as e:
        raise RuntimeError(f"Invalid file structure: {str(e)}")
    
    # Prepare columns to read (always include time)
    if columns_to_read is None:
        columns_to_read = list(column_mappings.keys())
    elif 'time' not in columns_to_read:
        columns_to_read = ['time'] + list(columns_to_read)

    # Map column names to positions
    cols_to_read = []
    final_column_names = []
    
    for short_name in columns_to_read:
        if short_name not in column_mappings:
            warnings.warn(f"Unknown column name: {short_name}")
            continue
            
        long_name = column_mappings[short_name]
        pos = None
        for desc, p in column_positions.items():
            if desc == long_name:
                pos = p
                break
                
        if pos is None:
            warnings.warn(f"Column not found in file: {long_name}")
            continue
                
        cols_to_read.append(pos)
        final_column_names.append(short_name)
    
    # Process data lines
    content_lines = file_content.splitlines()
    _, coldef_end_idx = find_section_boundaries(content_lines)
    
    data = []
    for line in content_lines[coldef_end_idx + 1:]:
        if re.match(r"\d{8}T\d{6}", line):  # Lines that start with timestamp
            fields = line.split()
            if len(fields) <= max(cols_to_read):
                continue
            row = [fields[i] for i in cols_to_read]
            data.append(row)
    
    if not data:
        raise ValueError("No valid data lines found in file")
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=final_column_names)
    
    # Convert numeric columns (all except time)
    numeric_cols = [col for col in df.columns if col != 'time']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert time and set index (if time column exists)
    
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%dT%H%M%S.%fZ')
    df.set_index('time', inplace=True)
    
    return df

def read_L2file(path, columns_to_read=None):
    """
    Read L2 files from either URL or local path.
    
    Args:
        path (str): URL or file path to the L2 file
        columns_to_read (list): List of columns to read
        
    Returns:
        pd.DataFrame: DataFrame containing the requested columns
    """
    # Determine if path is URL or local file
    parsed = urlparse(path)
    is_url = bool(parsed.scheme and parsed.netloc)
    
    print(f"Reading L2 file from {path}")
    # Get file content as string
    if is_url:
        file_string = download_txt_file_to_string(path)
    else:
        file_string = read_local_txt_to_string(path)
    
    # Process file content
    df = df_from_L2_string(file_string, columns_to_read)
    
    return df




if __name__ == '__main__':
    # Handle command line arguments
    if len(sys.argv) < 2:
        print("Usage: python L0_dataloaders.py <path> [column1 column2 ...]")
        sys.exit(1)
        
    path = sys.argv[1]
    columns_to_read = sys.argv[2:] if len(sys.argv) > 2 else ['time', 'routine_code']
    
    try:
        main_df, skipped_df = read_L0file(path, columns_to_read)
        print(f"\nSuccessfully loaded data with {len(main_df)} rows")
        print(f"Skipped {len(skipped_df)} lines")
        print("\nDataFrame head:")
        print(main_df.head())
        
        if not skipped_df.empty:
            print("\nSkipped lines summary:")
            print(skipped_df['reason'].value_counts())
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
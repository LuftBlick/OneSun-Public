import numpy as np
import pandas as pd
import scipy.signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import h5py

datapath= 'G:/Meine Ablage/LuftBlick/data/HR_database/'



def get_F0(wl_start):
    '''
    Load extraterrestrial solar spectrum from an HDF5 database.
    '''
    filename = f'HR_{wl_start}-{wl_start+20}nm.h5'
    print(f'loading {filename}')
    with h5py.File(datapath + filename, 'r') as file:
        dwl = 0.001
        wl_array = np.arange(wl_start, wl_start + 20 + dwl, dwl)
        
        F0_df = pd.DataFrame({'wl': wl_array})
        
        data = [d[0][0] * 1e-7 for d in file['F0_1']]
        F0_df['data'] = data
        
    return F0_df

def get_all_F0(wl_s, wl_e):
    '''
    Combine extraterrestrial solar spectra from multiple files into a single DataFrame.
    '''
    F0_all_df = pd.DataFrame()
    
    for wl_start in np.arange(wl_s, wl_e + 20, 20):
        F0_df = get_F0(wl_start)
        F0_all_df = pd.concat([F0_all_df, F0_df], ignore_index=True)
    
    return F0_all_df



def get_ods_gases(gas_table, gases_all, wl_start, T_r):
    '''
    get ods for all gases in gas_table for a given wavelength
    '''
    filename = f'HR_{wl_start}-{wl_start+20}nm.h5'
    with h5py.File(datapath + filename, 'r') as fi:
        dwl = 0.001
        wl_ar = np.arange(wl_start, wl_start + 20 + dwl, dwl)
        
        gas_keys = list(fi.keys())
        ods_df = pd.DataFrame({'wl': wl_ar})
        
        for gas in gases_all:
            print(gas)
            #gas_names = [s for s in gas_keys if gas in s]
            gas_names = [s for s in gas_keys if s.split('_')[0] == gas]

            if gas_names:
                print('found in file')
                gas_name = gas_names[0]

                data = list(fi[gas_name])
                nop = len(data[0][0])  # number of polynomials
                ps = [[d[0][n] for n in range(nop)] for d in data]

                T_ref = gas_table[gas_table['gas'] == gas]['Teff'].values[0]
                T = T_ref + T_r
                sf = 1e-7
                T_scale = 50.

                ods = [sum(p[n] * sf * ((T - T_ref) / T_scale) ** n for n in range(nop)) for p in np.array(ps)]
            else:
                ods = np.zeros(len(wl_ar))
            
            ods_df[gas] = ods
    
    return ods_df


def get_all_ods(gas_table, gases_all, wl_s, wl_e, T_r):
    ods_all_df = pd.DataFrame()
    
    for wl_start in np.arange(wl_s, wl_e + 20, 20):
        print(wl_start)
        ods_df = get_ods_gases(gas_table, gases_all, wl_start, T_r)
        ods_all_df = pd.concat([ods_all_df, ods_df], ignore_index=True)

    return ods_all_df






def convolute_and_interpolate_spectrum(df, wl_start, wl_end, dwl, slit_width_nm=0.5, slit_function='gaussian'):
    """
    Convolute a high-resolution spectrum with a slit function and interpolate onto a given wavelength grid.

    Returns a pandas DataFrame with convoluted and interpolated spectrum.
    """
    
    # Define the Gaussian and triangular slit functions
    def gaussian(x, mean, std):
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    def triangular(x, width):
        return np.where(np.abs(x) <= width, 1 - np.abs(x) / width, 0)
    
    # Generating slit function values
    slit_range = np.arange(-slit_width_nm*3, slit_width_nm*3, 0.001)
    if slit_function == 'gaussian':
        slit_values = gaussian(slit_range, 0, slit_width_nm / 2.355)
    elif slit_function == 'triangular':
        slit_values = triangular(slit_range, slit_width_nm)
    
    # Convolution
    convoluted_spectrum = scipy.signal.convolve(df['data'], slit_values, mode='same', method='auto') / sum(slit_values)
    
    # Interpolation to the given wavelength grid
    new_wl_grid = np.arange(wl_start, wl_end, dwl)
    interpolate_func = interp1d(df['wl'], convoluted_spectrum, kind='cubic', fill_value="extrapolate")
    interpolated_data = interpolate_func(new_wl_grid)
    
    # Creating the new DataFrame
    new_df = pd.DataFrame({'wl': new_wl_grid, 'data': interpolated_data})
    
    return new_df


def OD_convolution_and_interpolation(ods_df, wl_start, wl_end, dwl, slit_width_nm=0.5, slit_function='gaussian'):
    # Initialize the final DataFrame with the correct wavelength grid
    wl_grid = np.arange(wl_start, wl_end, dwl)
    final_df = pd.DataFrame({'wl': wl_grid})

    # Loop over each gas column in ods_df, except the 'wl' column
    for column in ods_df.columns[1:]:  # Skipping the 'wl' column
        # Extract the column as a Series with the wavelength column
        temp_df = pd.DataFrame({
            'wl': ods_df['wl'],
            'data': ods_df[column]
        })
        
        # Apply the convolute_and_interpolate_spectrum function
        convoluted_df = convolute_and_interpolate_spectrum(temp_df, wl_start, wl_end, dwl, slit_width_nm, slit_function)
        
        # Add the convoluted and interpolated data to the final DataFrame
        final_df[column] = convoluted_df['data']

    return final_df



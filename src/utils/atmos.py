import numpy as np
import pytz
from pysolar.solar import get_altitude
import pandas as pd


def get_airmasses(SZA, alt):
    '''
    Calculate airmasses for different types of scattering in the atmosphere.

    Parameters:
    SZA : Solar Zenith Angle in degrees
    alt : Altitude above sea level in km 

    Returns:
    m_ray : Rayleigh Scattering Airmass
    m_aero : Aerosol Scattering Airmass
    m_gas : Trace Gas (e.g., O3) Scattering Airmass
    '''
    alt = alt / 1000. # convert altitude from m to km
    # convert SZA from degrees to radians for use in numpy functions
    SZA_rad = np.radians(SZA)
    
    # Rayleigh Scattering Airmass (Kasten and Young, 1989)
    m_ray = 1 / (np.cos(SZA_rad) + 0.50572 * (96.07995 - SZA)**(-1.6364)) 

    # Trace Gas (e.g., O3) Scattering Airmass (Komhyr, 1989)
    R = 6371.229 # Earth radius in km
    h = 21 # Effective height of Ozone layer in km
    m_gas = (R + h) / np.sqrt((R + h)**2 - (R + alt)**2 * np.sin(SZA_rad)**2)

    # Aerosol Scattering Airmass (Kasten, 1965)
    m_aero = 1 / (np.cos(SZA_rad) + 0.0548 * (92.65 - SZA)**(-1.452))      
    
    return m_ray, m_aero, m_gas


def sun_earth_dist(doy):
    '''
    Compute normalized earth-sun distance.

    Based on formula from Paltridge and Platt, Radiative Processes in Climatology,1979
    Implemented by B. Schmid on 14.7.1995

    Parameters:
    doy : Day of the year

    Returns:
    f : Normalized earth-sun distance
    '''
    angle = 2 * np.pi * (doy - 1) / 365
    f = (1.00011 + 0.034221 * np.cos(angle) + 0.00128 * np.sin(angle) +
         0.000719 * np.cos(2 * angle) + 0.000077 * np.sin(2 * angle))
        
    return f



def get_sza(lat, lon, dt):
    '''
    get solar zenith angle (SZA)
    from pysolar module
    '''
    # Pysolar's get_altitude function expects latitude and longitude in degrees,
    # and the date-time in a timezone-aware format.
    dt = dt.replace(tzinfo=pytz.UTC)
    
    alt = get_altitude(lat, lon, dt)
    
    # Return solar zenith angle
    return 90 - alt


def get_tau_rayleigh(wl, alt):
    '''
    Compute Rayleigh Optical Depth according to 
    B.A.Bodhaine, J.Atmos.Ocean.Tech., 16, 1854 (1999)

    Parameters:
    wl : Wavelength in nm
    alt : Altitude in m

    Returns:
    tau_r : Rayleigh Optical Depth
    '''
    
    # Barometric formula for pressure calculation at given altitude
    p = 1013 * (1 - 6.5 / 288.15 * alt / 1000) ** 5.255 

    # Constants
    Avogadro = 6.0221367e23  # Avogadro's constant
    ma = 15.0556 * 0.00036 + 28.9595  # Mean molecular mass with 360ppm CO2
    g = 980.616  # Gravitational acceleration (cm/s^2)

    # Coefficients for the Rayleigh optical depth formula
    a = 1.0455996
    b = 341.29061
    c = 0.9023085
    d = 0.0027059889
    e = 85.968563
    
    wl = wl / 1000.  # Convert wavelength from nm to um
    # Rayleigh optical depth formula
    tau_r = (1e-28 * 1000 * Avogadro / (ma * g) * 
             (a - b * wl ** -2 - c * wl ** 2) / 
             (1 + d * wl ** -2 - e * wl ** 2))
    tau_r *= p
    
    return tau_r



def calculate_sza_am(df, lat, lon, alt):
    '''
    Calculate Solar Zenith Angle (SZA) and Air Mass (AM)

    Parameters:
    df : DataFrame whose index will be used as timestamps
    lat : Latitude of the location
    lon : Longitude of the location
    alt : Altitude of the location

    Returns:
    sza_am_df : DataFrame with SZA and AM as columns and same index as df
    '''
    sza_values = []
    am_gas_values = []
    am_ray_values = []
    am_aero_values = []


    for dt in df.index:
        dt_datetime = dt.to_pydatetime()
        sza = get_sza(lat, lon, dt_datetime)
        m_ray, m_aero, m_gas = get_airmasses(sza, alt)

        sza_values.append(sza)
        # Here I'm using m_gas for the air mass. 
        # Depending on the application, you may want to use m_ray or m_aero instead.
        am_gas_values.append(m_gas)
        am_ray_values.append(m_ray)  
        am_aero_values.append(m_aero)
        
    sza_am_df = pd.DataFrame({
        'sza': sza_values,
        'am_aero': am_aero_values,
        'am_ray': am_ray_values,
        'am_gas': am_gas_values
    }, index=df.index)

    return sza_am_df


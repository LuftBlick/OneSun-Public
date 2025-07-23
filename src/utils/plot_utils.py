import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates


def set_mpl_style(theme='light'):
    """
    Set matplotlib style based on selected theme.
    
    Args:
        theme (str): One of 'light', 'dark', 'darker'
    """
    # Define color schemes for different themes
    themes = {
        'white': {
            'axes_face': '1',    # very light gray background
            'fig_face': '1',
            'axes_edge': '1',     # dark gray axes
            'grid': '.82',         # light gray grid
            'text': '.2'           # dark gray text
        },
        'light': {
            'axes_face': '.95',    # very light gray background
            'fig_face': '.96',
            'axes_edge': '.95',     # dark gray axes
            'grid': '.85',         # light gray grid
            'text': '.2'           # dark gray text
        },
        'dark': {
            'axes_face': '.2',     # dark gray background
            'fig_face': '.19',
            'axes_edge': '.85',    # light gray axes
            'grid': '.3',          # darker gray grid
            'text': '.85'          # light gray text
        },
        'darker': {
            'axes_face': '.12',     # very dark gray background
            'fig_face': '.1',
            'axes_edge': '.8',     # very light gray axes
            'grid': '.2',          # very dark gray grid
            'text': '.8'           # very light gray text
        },
        'darker2': {
            'axes_face': '.11',     # very dark gray background
            'fig_face': '.12',
            'axes_edge': '.2',     # very light gray axes
            'grid': '.2',          # very dark gray grid
            'text': '.5'           # very light gray text
        }
    }
    
    if theme not in themes:
        raise ValueError(f"Theme must be one of {list(themes.keys())}")
    
    colors = themes[theme]
    
    # Set style parameters
    plt.style.use('default')  # Reset to default first
    
    params = {
        # Axes
        'axes.facecolor': colors['axes_face'],
        'axes.linewidth': 0.5,
        'axes.edgecolor': colors['axes_edge'],
        'axes.axisbelow': True,
        'axes.labelcolor': colors['text'],
        
        # Ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 0,
        'ytick.major.size': 0,
        'xtick.minor.size': 0,
        'ytick.minor.size': 0,
        'xtick.color': colors['text'],
        'ytick.color': colors['text'],
        
        # Grid
        'grid.color': colors['grid'],
        'grid.linewidth': 0.5,
        'grid.linestyle': '-',
        
        # Text
        'text.color': colors['text'],
        
        # Figure
        'figure.facecolor': colors['fig_face'],
        'figure.edgecolor': colors['axes_edge'],
        
        # Saving
        'savefig.facecolor': colors['fig_face'],
        'savefig.edgecolor': colors['axes_edge']
    }
    
    plt.rcParams.update(params)





def plot_pixel_timeseries(df,ncols=5):
    """Plot time series for specific pixels across days."""
    fig = plt.figure(figsize=(20,12))
    #ncols = 5
    w, h = .2, .15
    dw, dh = .02, .05
    

    # Convert datetime column to pandas datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df['datetime'])
    
  # Get cloudscreen mask
    clean_mask = dl0.cloudscreen_df(df, time_window='360s', pixel_idx=1000, lim=5e-2)
      

    # Get unique dates
    dts = np.unique(df.index.date)
    
    # Selected pixels to plot
    pixels = [2000, 1500, 1000, 500]
    colors = [(.1,.1,.3), (.1,.3,.8), (.2,.7,.9), (.7,.5,.7)]
    
    k = 0
    for day_value in dts:
        day_date = pd.Timestamp(day_value)
        day_df = df[df.index.date == day_value]
        day_mask = clean_mask[df.index.date == day_value]

        
        r, c = divmod(k, ncols)
        ax = fig.add_axes([c*(w+dw), 1-(r+1)*(h+dh), w, h])
        
        # plot filterwheel_pos2
        #ax.plot(day_df['filterwheel_pos1'].astype(int) * 1e4 ,'.',ms=1)
        #ax.plot(day_df['filterwheel_pos2'].astype(int) * 1e4 ,'.',ms=1)
        # Plot data for each selected pixel
        ms = 4
        for pixel, color in zip(pixels, colors):
            pixel_values = [spectrum[pixel] for spectrum in day_df['data']]

        #    # Extract the specified pixel from each spectrum
        #    pixel_values = [spectrum[pixel] for spectrum in day_df['data']]
        #    
        #    ax.plot(day_df.index, pixel_values, '.-', c=color, ms=ms, lw=.1)
        
            # Plot cloudy data
            cloudy_times = day_df.index[~day_mask]
            cloudy_values = [pixel_values[i] for i in range(len(pixel_values)) 
                            if day_df.index[i] in cloudy_times]
            if cloudy_values:
                ax.plot(cloudy_times, cloudy_values, '.-', c=(.7,.7,.7), 
                        alpha=0.5, ms=ms-1, lw=.1)
                
            # Plot clean data
            clean_times = day_df.index[day_mask]
            clean_values = [pixel_values[i] for i in range(len(pixel_values)) 
                            if day_df.index[i] in clean_times]
            if clean_values:
                ax.plot(clean_times, clean_values, '.-', c=color, 
                        alpha=1, ms=ms, lw=.1)
      
        # Set time limits and format
        start_time = day_date + pd.Timedelta(hours=2)
        end_time = day_date + pd.Timedelta(hours=21)
        ax.set_xlim(start_time, end_time)
        
        # Set x-ticks
        tick_hours = [6, 9, 12, 15, 18, 21]
        tick_times = [day_date + pd.Timedelta(hours=h) for h in tick_hours]
        ax.set_xticks(tick_times)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        
        ax.set_ylim(-1e3,5e4)

        # Add title
        ax.annotate(str(day_value), xy=(0,1.05), xycoords='axes fraction', 
                   ha='left', fontsize=12)
        
        ax.grid(True)
        k += 1
    
    #plt.show()
    return fig

def plot_all_spectra(df):
    """Plot all spectra from the data column overlaid."""
    ax = plt.figure(figsize=(10,6)).add_axes([0,0,1,.4])
    
    for spectrum in df['data']:
        if isinstance(spectrum, str):
            spectrum = eval(spectrum)  # Convert string to list if needed
        ax.plot(spectrum,'.-',lw = .1,ms=.1 ,alpha=0.1, color='blue')
    
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Intensity')
    
    return ax


def plot_prediction(L0, L1_i, L1_a):
    """Plot the predicted L1_a."""
    L1_a_np = L1_a.squeeze().cpu().numpy()
    L1_i_np = L1_i.squeeze().cpu().numpy()
    L0_np = L0.squeeze().cpu().numpy()[:2000]

    # Plot the predicted L1_a
    plt.figure(figsize=(10, 5))
    plt.plot(L0_np, label='L0')
    plt.plot(L1_i_np, label='L1_i')
    plt.plot(L1_a_np, label='L1_a')
    plt.title('Predicted L1_a')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Intensity')
    plt.legend()  # Add legend
    plt.show()
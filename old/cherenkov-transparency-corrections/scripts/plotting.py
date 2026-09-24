import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os

# Example colors
c1 = (5/255,5/255,153/255)
c2 = (102/255,0/255,204/255)
c3 = (255/255,51/255,204/255)
c4 = (204/255,0/255,0/255)
c5 = (255/255,225/255,0/255)
default_colors = [c1, c2, c3, c4, c5]

def params(n=15):
    '''
    Function to set standard parameters for matplotlib.

    Parameters:
    - n (int): Font size for matplotlib.

    '''
    plt.rcParams['font.size'] = n
    plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    plt.rcParams['axes.linewidth'] = 1.9
    plt.rcParams['figure.figsize'] = (13, 7)
    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['xtick.major.width'] = 1.8
    plt.rcParams['ytick.major.width'] = 1.8   
    plt.rcParams['lines.markeredgewidth'] = 2
    pd.set_option('display.max_columns', None)

def color_gradient(x, input_colors=default_colors):
    '''
    Function to create a color gradient of 5 colors in this case
    
    Parameters:
    - x (float): The value from 0 to 1 to assign a color
    - input_colors (list): List of tuples or list of strings, optional. The list of colors to use for the gradient. 
         Each color can be specified as a tuple of RGB values or as a string representing a named color. 
         Default is default_colors.
            
    Returns:
    - tuple: The RGB values for the color to assign
    
    Raises:
    - ValueError: If the input x is not a float in the range [0, 1]
    '''
    size = len(input_colors)
    size_bins = size - 1 
    
    COLORS = []
    for i in range(size):
        
        if type(input_colors[i]) == str:
            c = colors.to_rgba(input_colors[i])
        else:
            c = input_colors[i]
        
        COLORS.append(c)
    
    try:
        x = float(x)
    except ValueError:
        raise ValueError(f'Input {x} should be a float in range [0 , 1]')
        
    if x > 1 or x < 0:
        raise ValueError(f'Input {x} should be in range [0 , 1]')
    
    for i in range(size_bins):
        if x >= i/size_bins and x <= (i+1)/size_bins:
            xeff = x - i/size_bins
            r = COLORS[i][0] * (1 - size_bins * xeff) + COLORS[i+1][0] * size_bins * xeff
            g = COLORS[i][1] * (1 - size_bins * xeff) + COLORS[i+1][1] * size_bins * xeff
            b = COLORS[i][2] * (1 - size_bins * xeff) + COLORS[i+1][2] * size_bins * xeff
            
    return (r, g, b)

def get_colors_multiplot(array, input_colors=default_colors, range=None, logscale=False):
    """
    Returns a list of colors corresponding to each element in the input array.
    
    Parameters:
    - array: list or array-like object containing the values for which colors are needed.
    - input_colors: list of colors to choose from. Default is default_colors.
    - ran: tuple (min, max) specifying the range of values. Default is None, in which case the 
            minimum and maximum values of the array are used.
    
    Returns:
    - output_colors: list of colors corresponding to each element in the input array.
    """
    if logscale:
        array = np.log10(array)
    
    # getting the color of each run
    output_colors = []
   
    if range != None:
        m, M = range[0], range[1]
    else:
        m, M = min(array), max(array)
    
    for i in np.arange(len(array)):
        
        if array[i] > M:
            output_colors.append(color_gradient(1, input_colors))
        elif array [i] < m:
            output_colors.append(color_gradient(0, input_colors))
        else:
            normalized_value = (array[i] - m) / (M - m)
            output_colors.append(color_gradient(normalized_value, input_colors))   
    
    return output_colors


def transparent_cmap(cmap, ranges=[0,1]):
    '''
    Returns a colormap object tuned to transparent.

    Parameters:
    - cmap (str): The name of the base colormap.
    - ranges (list, optional): The range of transparency values. Defaults to [0, 1].

    Returns:
    - matplotlib.colors.LinearSegmentedColormap: The transparent colormap object.
    '''

    ncolors = 256
    color_array = plt.get_cmap(cmap)(range(ncolors))
    color_array[:,-1] = np.linspace(*ranges, ncolors)

    # building the colormap
    return colors.LinearSegmentedColormap.from_list(name='cmap', colors=color_array)


def create_cmap_from_colors(input_colors):
    '''
    Create a colormap given an array of colors
    
    Parameters:
    - colors (list): List of colors to create the colormap from
    
    Returns:
    - matplotlib.colors.LinearSegmentedColormap: The created colormap
    '''    
    return colors.LinearSegmentedColormap.from_list('',  input_colors)

default_cmap = create_cmap_from_colors(default_colors)
cylon = create_cmap_from_colors([(68/255,0,0), (216/255,0,72/255), (1,134/255,61/255), (1,239/255,224/255)])

def plot_colorbar(fig, ax, array, cmap, label="", discrete_int=False, use_scientific_notation=True):
    """
    Add a colorbar to a matplotlib figure.

    Parameters:
    - fig: The matplotlib figure object.
    - ax: The matplotlib axes object where the colorbar will be added.
    - array: The array of values used to determine the color of each element in the colorbar.
    - cmap: The colormap used to map the values in the array to colors.
    - label: The label for the colorbar.
    - discrete_int: If the colorbar is set to be used for a discrete colormap for INTEGER values
    - use_scientific_notation: (Boolean) In case we want to disable scientific notation in axis

    Returns:
    None
    """
    delta_int = 0.5 if discrete_int else 0.0
    norm = mpl.colors.Normalize(vmin=min(array)-delta_int, vmax=max(array)+delta_int)
    
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=label)
    
    if not use_scientific_notation:
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

def get_cmap_colors(array, cmap):
    """
    Get normalized values and colors from an array using a specified colormap.

    Parameters:
    - array (numpy.ndarray): The input array.
    - cmap (matplotlib.colors.Colormap): The colormap to use.

    Returns:
    - tuple: A tuple containing the normalization object and the array of colors.
    """

    norm   = mpl.colors.Normalize(vmin=np.min(array), vmax=np.max(array))
    output_colors = mpl.cm.ScalarMappable(norm, cmap).to_rgba(array)    
    
    return norm, output_colors

def create_discrete_cmap_from_colors(num_discretization=10, input_colors=default_colors):
    """
    Creating a colormap from a list of colors. Setting the number of discrete steps, and the input colors.
    
    Parameters:
    - num_discretixation (int): Number of discrete steps
    - input_colors (colors array): Array of colors, by default using predetermined of this script.
    """
    
    cmap_smooth = create_cmap_from_colors(input_colors)
    midpoints = np.linspace(0, 1, num_discretization)
    cmap_discrete = cmap_smooth(midpoints)
    
    cmap = colors.LinearSegmentedColormap.from_list(
        f"discretized_{num_discretization}", 
        cmap_discrete, 
        N = num_discretization
    )
    return cmap

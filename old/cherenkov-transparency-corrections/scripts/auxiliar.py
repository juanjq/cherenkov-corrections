import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.colors as colors
from scipy.stats import chi2

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# Some symbols to add easily to text
sigma = "$\sigma$"
mu    = "$\mu$"


def find_files_in_dir(directory):
    """
    Find all files in a directory and its subdirectories.

    Args:
        directory (str): The directory to search.

    Returns:
        list: A list of file paths.
    """
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files]


def createdir(path):
    """
    Create a directory if it doesn't exist.

    Args:
        path (str): The path of the directory to be created.

    Returns:
        None
    """
    # checking if exists first
    if not os.path.exists(path):
        os.makedirs(os.path.join(path), exist_ok=True)


def find_nearest_value(array, value):
    """
    Finding the nearest value inside an array.

    Parameters:
    array (list or numpy.ndarray): The input array.
    value (int or float): The value to find the nearest integer to.

    Returns:
    int or float: The nearest value to the given value in the array.
    """
    idx = (np.abs(np.array(array) - value)).argmin()
    return array[idx]


def move_files(source_folder, destination_folder):
    '''
    Function to move files from one directory to another

    Parameters:
        source_folder (str): The path of the source folder containing the files to be moved.
        destination_folder (str): The path of the destination folder where the files will be moved to.
    '''

    # iterating over all files inside the folder
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)

        # checking if is a file or a folder
        if os.path.isfile(source_path):
            destination_path = os.path.join(destination_folder, filename)

            # moving file by file
            shutil.move(source_path, destination_path)


def delete_directory(directory_path):
    '''
    Deletes a directory if it exists.

    Parameters:
        directory_path (str): The path of the directory to be deleted.

    Returns:
        None

    Raises:
        OSError: If an error occurs while deleting the directory.
    '''
    
    try:
        os.rmdir(directory_path)
        logger.info(f"Directory '{directory_path}' deleted successfully.")
        
    except OSError as error:
        logger.error(f"Error deleting directory '{directory_path}': {error}")


def sort_based(x_array, ref_array):
    """
    Sorts the array ref_array in ascending order and rearranges the array x_array based on the sorted ref_array values.

    Parameters:
    x_array (array-like): The array to be rearranged based on ref_array.
    ref_array (array-like): The reference array used for sorting.

    Returns:
    tuple: A tuple containing two arrays. The first array is the sorted ref_array, and the second array is x_array rearranged based on the sorted ref_array values.
    """
    return np.sort(ref_array), np.array([x for ref, x in sorted(zip(ref_array, x_array))])
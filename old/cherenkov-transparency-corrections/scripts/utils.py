import numpy as np
import json
from datetime import datetime, timedelta

def merge_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries with nested structures.

    Args:
    - dict1: First dictionary
    - dict2: Second dictionary

    Returns:
    - Merged dictionary
    """
    merged = dict1.copy()

    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # If both values are dictionaries, recursively merge them
            merged[key] = merge_dicts(merged[key], value)
        else:
            # Otherwise, just update or add the key-value pair
            merged[key] = value
    return merged

def read_json_files(file_paths):
    """Read multiple JSON files and return a list of their contents."""
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data.append(json.load(file))
    return data

def merge_json_data(data_list):
    """Merge a list of JSON data dictionaries."""
    merged_data = {}
    for data in data_list:
        # Assuming data is a dictionary; for lists, use `merged_data.extend(data)`
        merged_data.update(data)  
    return merged_data

def modify_json_data(target_dict, change_dict):
    """
    Recursively update the target_dict with values from change_dict.
    Changes are applied to only the topmost level of the target_dict.
    
    Parameters:
    target_dict (dict): The original dictionary to be updated.
    change_dict (dict): The dictionary containing the changes to be applied.
    
    Returns:
    dict: The updated dictionary with changes applied.
    """
    # Iterate through the change_dict to apply changes
    for key, value in change_dict.items():
        # If the value is a dictionary, recurse into it
        if isinstance(value, dict):
            # Create the key if it doesn't exist or update if it does exist
            if key not in target_dict:
                target_dict[key] = {}
            # Recursively call update_dict to handle nested structures
            modify_json_data(target_dict[key], value)
        else:
            # If it's not a dictionary, directly update or add the key
            target_dict[key] = value
    return target_dict

def write_json_file(data, output_file):
    """Write the modified JSON data to an output file."""
    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)

        
def gps_to_utc(gps_seconds):
    # Define GPS epoch
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    # Convert GPS time to UTC (subtract leap seconds)
    utc_time = gps_epoch + timedelta(seconds=gps_seconds - 18)  # 18 leap seconds
    return utc_time



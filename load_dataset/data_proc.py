""" 
This module contains functions for processing data visualization images.
"""
import numpy as np

def load_file_to_nparr(f):
    try:
        return np.load(f)
    except:
        print(f"File {f} does not exists.")

def dataset_loader(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    return arr


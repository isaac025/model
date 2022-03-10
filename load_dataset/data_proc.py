""" 
This module contains functions for processing data visualization images.
"""
import torch 
import numpy as np

def load_file_to_nparr(f):
    try:
        return np.load(f)
    except:
        print(f"File {f} does not exists.")

def dataset_loader(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    (x,y) = np.shape(arr)

    x0 = 0
    x1 = 1

    data = arr[x1:x]
    targets = arr[x0]

    return (data,targets)


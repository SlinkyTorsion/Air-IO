import numpy as np
import matplotlib.pyplot as plt
import pypose as pp
from scipy.spatial.transform import Rotation, Slerp
import os
import pickle

def load_airimu_data(pickle_path, dataset_name=None):
    """
    Loads rotation data from a pickle file produced by inference_motion.py
    
    Args:
        pickle_path: Path to the pickle file containing rotation data
        dataset_name: Optional specific dataset name to extract from the pickle
        
    Returns:
        dict: A dictionary containing the loaded rotation data and timing information
    """
    try:
        with open(pickle_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        print(f"Successfully loaded data with keys: {list(loaded_data.keys())}")
            
        return loaded_data
    
    except FileNotFoundError:
        print(f"Error: File {pickle_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test AirIMU data loading and analysis')
    parser.add_argument('--pickle_path', type=str, required=True,
                        help='Path to the pickle file containing AirIMU data')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Optional specific dataset name to extract from the pickle')
    
    args = parser.parse_args()
    
    # Load the data
    data = load_airimu_data(args.pickle_path, args.dataset_name)
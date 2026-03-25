import pandas as pd
import numpy as np
import streamlit as st

def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def standardize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def validate_binary_target(y):
    unique_targets = np.unique(y)
    if len(unique_targets) != 2:
        return False, unique_targets, y
    
    if not set(unique_targets).issubset({0, 1}):
        # Map values to 0 and 1
        y_mapped = np.where(y == unique_targets[0], 0, 1)
        return True, unique_targets, y_mapped
    return True, unique_targets, y

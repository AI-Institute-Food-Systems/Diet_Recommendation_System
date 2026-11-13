"""Seeding and input checks used by `cli/evaluate.py`."""

import numpy as np
from typing import Dict, Any
import warnings


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def validate_data_contracts(data: Dict[str, Any]) -> bool:
    """
    Validate that loaded data meets the expected contracts.
    
    Args:
        data: Loaded data dictionary
        
    Returns:
        True if validation passes, False otherwise
    """
    required_keys = ['real_meals', 'generated_meals', 'W', 'nutrient_names', 'RDI']
    
    for key in required_keys:
        if key not in data:
            warnings.warn(f"Missing required data: {key}")
            return False
    
    # Validate shapes
    if 'W' in data and 'nutrient_names' in data:
        if data['W'].shape[1] != len(data['nutrient_names']):
            warnings.warn("Nutrient matrix and names dimension mismatch")
            return False
    
    if 'RDI' in data and 'nutrient_names' in data:
        if len(data['RDI']) != len(data['nutrient_names']):
            warnings.warn("RDI and nutrient names dimension mismatch")
            return False
    
    # Validate meal data
    if 'real_meals' in data and 'generated_meals' in data:
        real_meals = data['real_meals']
        gen_meals = data['generated_meals']
        
        if set(real_meals.keys()) != set(gen_meals.keys()):
            warnings.warn("Meal type mismatch between real and generated data")
            return False
        
        for meal_type in real_meals.keys():
            if real_meals[meal_type].shape[1] != gen_meals[meal_type].shape[1]:
                warnings.warn(f"Feature dimension mismatch for {meal_type}")
                return False
    
    return True

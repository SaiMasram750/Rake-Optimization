import pandas as pd
import numpy as np

def calculate_demand_probability(weight):
    """
    Calculate probability distribution across demand categories based on weight.
    Returns probabilities for High, Medium, and Low demand categories.
    """
    # Using sigmoid function to create smooth probability transitions
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Calculate base probabilities using weight thresholds
    high_prob = sigmoid((weight - 25) / 5)  # Center point at 25 tons
    low_prob = sigmoid((15 - weight) / 5)   # Center point at 15 tons
    
    # Medium probability is highest when weight is between thresholds
    medium_prob = 1 - (high_prob + low_prob) / 2
    
    # Normalize probabilities to sum to 1
    total = high_prob + medium_prob + low_prob
    probabilities = {
        'High': round((high_prob / total) * 100, 2),
        'Medium': round((medium_prob / total) * 100, 2),
        'Low': round((low_prob / total) * 100, 2)
    }
    
    return probabilities

def predict_demand_category(df):
    """
    Predict demand category and calculate probabilities for each category.
    Returns DataFrame with demand category and probability distributions.
    """
    # Calculate probabilities for each row
    probability_dicts = df['weight'].apply(calculate_demand_probability)
    
    # Extract individual probability columns
    df['high_demand_prob'] = probability_dicts.apply(lambda x: x['High'])
    df['medium_demand_prob'] = probability_dicts.apply(lambda x: x['Medium'])
    df['low_demand_prob'] = probability_dicts.apply(lambda x: x['Low'])
    
    # Determine the demand category based on highest probability
    df['demand_category'] = df[['high_demand_prob', 'medium_demand_prob', 'low_demand_prob']].idxmax(axis=1).map({
        'high_demand_prob': 'High',
        'medium_demand_prob': 'Medium',
        'low_demand_prob': 'Low'
    })
    
    return df

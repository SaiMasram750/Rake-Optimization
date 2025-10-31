import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache

class DemandPredictor:
    """
    Enhanced demand prediction system with multi-factor analysis.
    Uses weight, material type, temporal patterns, and historical trends.
    """
    
    def __init__(self):
        # Material priority weights (high priority materials = higher demand)
        self.material_weights = {
            'steel': 1.3,
            'iron ore': 1.2,
            'coal': 1.1,
            'limestone': 0.9,
            'dolomite': 0.85,
            'default': 1.0
        }
        
        # Seasonal demand multipliers (if date column available)
        self.seasonal_multipliers = {
            'Q1': 1.1,  # Jan-Mar: High demand
            'Q2': 0.95, # Apr-Jun: Moderate
            'Q3': 1.05, # Jul-Sep: Moderate-High
            'Q4': 1.2   # Oct-Dec: Peak demand
        }
        
        # Weight thresholds for better categorization
        self.weight_thresholds = {
            'high': 30,      # > 30 tons
            'medium_high': 20,  # 20-30 tons
            'medium_low': 12,   # 12-20 tons
            'low': 12        # < 12 tons
        }
        
        # Cache for repeated calculations
        self._probability_cache = {}
    
    @lru_cache(maxsize=1000)
    def _calculate_weight_score(self, weight):
        """
        Calculate normalized weight score using piecewise sigmoid.
        Cached for performance with repeated weights.
        """
        def sigmoid(x, center, steepness=0.3):
            return 1 / (1 + np.exp(-steepness * (x - center)))
        
        # Multi-stage sigmoid for better separation
        if weight >= self.weight_thresholds['high']:
            score = 0.85 + 0.15 * sigmoid(weight, 35, 0.1)
        elif weight >= self.weight_thresholds['medium_high']:
            score = 0.50 + 0.35 * sigmoid(weight, 25, 0.2)
        elif weight >= self.weight_thresholds['medium_low']:
            score = 0.25 + 0.25 * sigmoid(weight, 16, 0.3)
        else:
            score = 0.10 + 0.15 * sigmoid(weight, 8, 0.4)
        
        return np.clip(score, 0, 1)
    
    def _get_material_factor(self, material):
        """Get material priority factor."""
        if pd.isna(material):
            return self.material_weights['default']
        
        material_lower = str(material).lower().strip()
        
        # Exact match
        if material_lower in self.material_weights:
            return self.material_weights[material_lower]
        
        # Partial match
        for key, value in self.material_weights.items():
            if key in material_lower or material_lower in key:
                return value
        
        return self.material_weights['default']
    
    def _get_temporal_factor(self, date_val):
        """Calculate temporal demand factor based on date."""
        if pd.isna(date_val):
            return 1.0
        
        try:
            if isinstance(date_val, str):
                date_obj = pd.to_datetime(date_val)
            else:
                date_obj = date_val
            
            # Determine quarter
            quarter = f'Q{(date_obj.month - 1) // 3 + 1}'
            return self.seasonal_multipliers.get(quarter, 1.0)
        except:
            return 1.0
    
    def _calculate_urgency_factor(self, priority=None, delivery_date=None):
        """
        Calculate urgency factor based on priority and delivery timeline.
        """
        urgency = 1.0
        
        # Priority-based urgency
        if not pd.isna(priority):
            priority_map = {
                'urgent': 1.3,
                'high': 1.2,
                'normal': 1.0,
                'low': 0.8
            }
            priority_str = str(priority).lower().strip()
            urgency *= priority_map.get(priority_str, 1.0)
        
        # Time-based urgency (if delivery date available)
        if not pd.isna(delivery_date):
            try:
                delivery = pd.to_datetime(delivery_date)
                today = pd.Timestamp.now()
                days_until = (delivery - today).days
                
                if days_until <= 7:
                    urgency *= 1.25  # Very urgent
                elif days_until <= 14:
                    urgency *= 1.15  # Urgent
                elif days_until <= 30:
                    urgency *= 1.05  # Moderate urgency
            except:
                pass
        
        return urgency
    
    def calculate_demand_probability(self, weight, material=None, date=None, 
                                    priority=None, delivery_date=None):
        """
        Calculate sophisticated probability distribution with multiple factors.
        """
        # Create cache key
        cache_key = (weight, material, str(date), priority, str(delivery_date))
        if cache_key in self._probability_cache:
            return self._probability_cache[cache_key]
        
        # Base weight score (0 to 1)
        weight_score = self._calculate_weight_score(weight)
        
        # Material factor (0.8 to 1.3)
        material_factor = self._get_material_factor(material)
        
        # Temporal factor (0.9 to 1.2)
        temporal_factor = self._get_temporal_factor(date)
        
        # Urgency factor (0.8 to 1.3)
        urgency_factor = self._calculate_urgency_factor(priority, delivery_date)
        
        # Combined score with weighted factors
        combined_score = (
            weight_score * 0.50 +           # Weight is most important
            material_factor * 0.25 +         # Material priority
            temporal_factor * 0.15 +         # Seasonal patterns
            urgency_factor * 0.10            # Urgency
        ) / 1.0
        
        # Normalize to 0-1 range
        combined_score = np.clip(combined_score, 0, 1)
        
        # Calculate probabilities using enhanced distribution
        if combined_score >= 0.7:
            # High demand dominant
            high_prob = 0.60 + (combined_score - 0.7) * 1.33
            medium_prob = 0.30 - (combined_score - 0.7) * 0.67
            low_prob = 0.10 - (combined_score - 0.7) * 0.33
        elif combined_score >= 0.4:
            # Medium demand dominant
            high_prob = 0.20 + (combined_score - 0.4) * 1.33
            medium_prob = 0.60 + (combined_score - 0.4) * 0.0
            low_prob = 0.20 - (combined_score - 0.4) * 0.33
        else:
            # Low demand dominant
            high_prob = 0.10 + combined_score * 0.25
            medium_prob = 0.30 + combined_score * 0.75
            low_prob = 0.60 - combined_score * 1.0
        
        # Ensure probabilities are valid
        high_prob = np.clip(high_prob, 0, 1)
        medium_prob = np.clip(medium_prob, 0, 1)
        low_prob = np.clip(low_prob, 0, 1)
        
        # Normalize to sum to 1
        total = high_prob + medium_prob + low_prob
        if total > 0:
            probabilities = {
                'High': round((high_prob / total) * 100, 2),
                'Medium': round((medium_prob / total) * 100, 2),
                'Low': round((low_prob / total) * 100, 2)
            }
        else:
            probabilities = {'High': 33.33, 'Medium': 33.33, 'Low': 33.34}
        
        # Cache the result
        self._probability_cache[cache_key] = probabilities
        
        return probabilities
    
    def predict_demand_category(self, df):
        """
        Predict demand category with enhanced feature extraction.
        Vectorized operations for better performance.
        """
        # Create copy to avoid modifying original
        df_result = df.copy()
        
        # Validate required columns
        if 'weight' not in df_result.columns:
            raise ValueError("DataFrame must contain 'weight' column")
        
        # Extract optional columns if available
        material_col = 'material' if 'material' in df_result.columns else None
        date_col = 'date' if 'date' in df_result.columns else 'order_date' if 'order_date' in df_result.columns else None
        priority_col = 'priority' if 'priority' in df_result.columns else None
        delivery_col = 'delivery_date' if 'delivery_date' in df_result.columns else None
        
        # Calculate probabilities for each row (vectorized where possible)
        results = []
        for idx, row in df_result.iterrows():
            material = row[material_col] if material_col else None
            date = row[date_col] if date_col else None
            priority = row[priority_col] if priority_col else None
            delivery = row[delivery_col] if delivery_col else None
            
            probs = self.calculate_demand_probability(
                weight=row['weight'],
                material=material,
                date=date,
                priority=priority,
                delivery_date=delivery
            )
            results.append(probs)
        
        # Convert results to DataFrame columns
        prob_df = pd.DataFrame(results)
        df_result['high_demand_prob'] = prob_df['High']
        df_result['medium_demand_prob'] = prob_df['Medium']
        df_result['low_demand_prob'] = prob_df['Low']
        
        # Determine demand category based on highest probability
        df_result['demand_category'] = df_result[[
            'high_demand_prob', 'medium_demand_prob', 'low_demand_prob'
        ]].idxmax(axis=1).map({
            'high_demand_prob': 'High',
            'medium_demand_prob': 'Medium',
            'low_demand_prob': 'Low'
        })
        
        # Add confidence score (max probability as confidence)
        df_result['confidence_score'] = df_result[[
            'high_demand_prob', 'medium_demand_prob', 'low_demand_prob'
        ]].max(axis=1)
        
        # Add risk flag for low confidence predictions
        df_result['low_confidence_flag'] = df_result['confidence_score'] < 50
        
        return df_result
    
    def get_feature_importance(self):
        """Return feature importance weights used in prediction."""
        return {
            'weight': 0.50,
            'material_type': 0.25,
            'temporal_patterns': 0.15,
            'urgency': 0.10
        }
    
    def clear_cache(self):
        """Clear probability cache."""
        self._probability_cache.clear()
        self._calculate_weight_score.cache_clear()


# Backward compatibility functions
_predictor_instance = DemandPredictor()

def calculate_demand_probability(weight):
    """
    Backward compatible function for simple weight-based prediction.
    """
    return _predictor_instance.calculate_demand_probability(weight)

def predict_demand_category(df):
    """
    Backward compatible function using enhanced predictor.
    """
    return _predictor_instance.predict_demand_category(df)


# Performance benchmarking function
def benchmark_predictor(df, iterations=1000):
    """
    Benchmark the predictor performance.
    """
    import time
    
    predictor = DemandPredictor()
    
    start_time = time.time()
    for _ in range(iterations):
        result = predictor.predict_demand_category(df)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    rows_per_second = len(df) / avg_time
    
    return {
        'avg_time_per_prediction': f"{avg_time*1000:.2f} ms",
        'rows_per_second': f"{rows_per_second:.2f}",
        'total_iterations': iterations
    }


if __name__ == "__main__":
    # Example usage and testing
    sample_data = pd.DataFrame({
        'order_id': ['O1', 'O2', 'O3', 'O4', 'O5'],
        'weight': [35, 22, 15, 8, 28],
        'material': ['steel', 'coal', 'limestone', 'dolomite', 'iron ore'],
        'priority': ['urgent', 'normal', 'high', 'low', 'urgent'],
        'date': pd.date_range('2024-01-01', periods=5)
    })
    
    predictor = DemandPredictor()
    result = predictor.predict_demand_category(sample_data)
    
    print("Enhanced Demand Prediction Results:")
    print(result[['order_id', 'weight', 'material', 'demand_category', 
                  'confidence_score', 'low_confidence_flag']])
    print(f"\nFeature Importance: {predictor.get_feature_importance()}")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_cholera_data():
    """Generate sample cholera data for testing"""
    
    # Generate dates for the last 2 years
    start_date = datetime.now() - timedelta(days=730)
    dates = [start_date + timedelta(days=i) for i in range(730)]
    
    # Generate realistic cholera case patterns
    np.random.seed(42)
    
    # Base seasonal pattern (higher in rainy season)
    seasonal_pattern = np.sin(np.linspace(0, 4*np.pi, 730)) * 20 + 30
    
    # Add trend
    trend = np.linspace(0, 10, 730)
    
    # Add random noise
    noise = np.random.normal(0, 8, 730)
    
    # Combine patterns
    cases = np.maximum(0, seasonal_pattern + trend + noise)
    cases = np.round(cases).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'cases': cases,
        'location': ['Region_A'] * 365 + ['Region_B'] * 365,
        'population': [100000] * 730,
        'temperature': np.random.normal(28, 5, 730),
        'rainfall': np.maximum(0, np.random.normal(50, 30, 730))
    })
    
    return df

if __name__ == "__main__":
    # Generate and save sample data
    sample_data = generate_sample_cholera_data()
    sample_data.to_csv('sample_cholera_data.csv', index=False)
    print("Sample data generated: sample_cholera_data.csv")
    print(f"Data shape: {sample_data.shape}")
    print("\nFirst few rows:")
    print(sample_data.head())

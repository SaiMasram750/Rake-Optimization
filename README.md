# Railway Rake Optimization System

A Python-based optimization system for railway rake formation using Streamlit and PuLP.

## Features

- Optimizes rake formation based on order weights and wagon capacities
- ML-based demand prediction with probability distributions
- Interactive web interface using Streamlit
- Wagon utilization metrics and warnings
- Downloadable optimization results
- Real-time visualization of assignments

## Data Files

- `order.csv`: Contains order details (ID, material, weight)
- `wagons.csv`: Contains wagon capacities
- `yard.csv`: Contains yard constraints

## How to Run

1. Install dependencies:
```bash
pip install streamlit pandas pulp numpy
```

2. Run the application:
```bash
streamlit run app.py
```

## Components

- `app.py`: Main Streamlit web interface
- `optimizer.py`: Core optimization logic using PuLP
- `ml_rules.py`: ML-based demand prediction system
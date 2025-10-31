## 🚆 Railway Rake Optimization System

A Python-based optimization system for railway rake formation using Streamlit and PuLP, now with a polished UI and smart analytics.

---

## 🎯 Key Features

✅ **AI-Powered Optimization**: Smart rake formation algorithms  
✅ **Real-time Analytics**: Monitor performance metrics  
✅ **Demand Prediction**: ML-based demand categorization  
✅ **Capacity Management**: Track wagon and yard utilization  
✅ **Export Reports**: Download detailed optimization plans  
✅ **Interactive UI**: User-friendly interface with sidebar navigation

---

## 🧭 Navigation

- **Home**  
- **Data Upload**  
- **View Data**  
- **Optimization**  
- **Analytics**

---

## 📁 Data Files

- `order.csv`: Contains order details (ID, material, weight)  
- `wagons.csv`: Defines wagon capacities  
- `yard.csv`: Contains yard constraints

The repository includes these sample CSV files in the project root for quick testing and demonstration. The app also provides a `Data Upload` page where users can upload their own CSV files to replace the defaults at runtime.

---

## 🚀 How to Run

1. **Install dependencies**:
```powershell
pip install streamlit pandas pulp numpy streamlit-option-menu
```

2. **Run the application**:
```powershell
streamlit run app.py
```

---

## 🧩 Components

- `app.py`: Main Streamlit web interface with navigation  
- `optimizer.py`: Core optimization logic using PuLP  
- `ml_rules.py`: ML-based demand prediction system  
- `requirements.txt`: Dependency list  
- `Procfile`: Deployment configuration

---

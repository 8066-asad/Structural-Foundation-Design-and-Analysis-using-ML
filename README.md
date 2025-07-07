# Structural Foundation Design and Analysis using ML
### Comprehensive README for Structural Foundation Design ML Models

#### 📌 Project Overview
This repository contains machine learning models for optimizing the design of two types of foundations:
1. **Raft Foundations** - Predicts settlement, punching shear, and bearing pressure
2. **Isolated Footings** - Predicts dimensions and reinforcement details

The models use advanced regression techniques to provide accurate structural design recommendations based on input parameters.

---

### 📋 Dependencies
Install required packages:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl joblib
```

---

### 🏗️ Raft Foundation Model

#### 📂 Data Preparation
- **Input Features**:
  ```python
  ['Number of Columns', 'Area Of Raft (m^2)', 'Column Area (m^2)',
   "Compressive strength of Concrete Fc' (Mpa)", 'Concrete Unit Weight (kN/m^3)',
   'Subgrade Modulus kN/m/m^2', 'Maximum Axial Load on Column in kN',
   'Total Axial load on Column (kN)', 'Thickness of Raft (mm)']
  ```
- **Output Targets**:
  ```python
  ['Settlement (mm)', 'Punching Shear Value', 'Bearing Pressure (kPa)']
  ```

#### 🤖 Model Architecture
```python
# Models tested:
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "XGBoost": XGBRegressor(n_estimators=100)
}

# Best performing model selected automatically
```

#### 📊 Evaluation Metrics
For each output:
- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

#### 💾 Model Saving
```python
joblib.dump(model, 'raft_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

---

### 🧱 Isolated Footing Model

#### 📂 Data Preparation
- **Input Features**:
  ```python
  ['Allowable Soil Bearing Pressure (Psf)', 
   'Compressive Strength of concrete (psi)',
   'Dead Load (kips)', 'Live Load (Kips)',
   'Yield Strength of Steel (Psi)',
   'Depth of footing Below Grade (ft)',
   'Unit Weight of Soil (pcf)',
   'Unit Weight of Concrete (lbs/ft^3)',
   'Width of Column (ft)',
   'Preferred Bar Number']
  ```
- **Output Targets**:
  ```python
  ['Footing Length (ft)', 'Footing Width (ft)', 
   'Number of Bars', 'Area of Steel Provided (in^2)',
   'Spacing (inches)']
  ```

#### 🤖 Model Architecture
- **Linear Regression** for most outputs
- **XGBoost** specifically for spacing prediction

#### 📊 Evaluation Metrics
- R² Score (as percentage)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

#### 🎯 Special Output Handling
```python
# Rounding rules:
footing_length = round(predicted_length * 4) / 4   # Nearest 0.25 ft
number_of_bars = int(np.ceil(predicted_bars))      # Round UP
spacing = np.floor(predicted_spacing * 4) / 4      # Round DOWN to 0.25
```

---

### 🚀 How to Use
1. **Data Upload**:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

2. **Run Prediction**:
   ```python
   # Raft Foundation
   user_input = [val1, val2, ...]  # 9 input values
   prediction = raft_model.predict(scaler.transform([user_input]))
   
   # Isolated Footing
   input_features = [...]  # 10 features
   predicted = model.predict(scaled_input)
   ```

3. **Save/Load Models**:
   ```python
   import joblib
   joblib.dump(model, 'foundation_model.pkl')
   loaded_model = joblib.load('foundation_model.pkl')
   ```

---

### 📈 Performance Highlights
| Model Type         | Best Algorithm | Avg R² Score |
|--------------------|----------------|--------------|
| Raft Foundation    | XGBoost        | 92.4%        |
| Isolated Footing   | Linear Regression + XGBoost | 88.7% |

---

### 🧮 Engineering Formulas
Key calculations used in preprocessing:
```python
# Reinforcement spacing rounding
def round_spacing(val):
    return int(math.floor(val / 5.0) * 5

# Footing dimension rounding
footing_length = round(predicted_value * 4) / 4
```

---

### 💡 Recommendations for Improvement
1. Add cross-validation for more robust evaluation
2. Implement hyperparameter tuning for tree-based models
3. Include uncertainty quantification in predictions
4. Develop web interface for easier input

---


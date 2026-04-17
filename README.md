https://youtu.be/U9ab-CbA6e0. >≤ my vidoe representation 



# 🚗 Car Price Prediction — End-to-End Machine Learning Pipeline

> Predicting used car prices with **96.47% accuracy** using an explainable, production-ready ML pipeline built on Databricks.

---

## 📌 Project Overview

This project builds a complete, end-to-end machine learning pipeline to predict car prices for a used car dealership. Starting from raw data, the pipeline covers every stage — cleaning, exploratory analysis, feature engineering, model training, AutoML comparison, and full explainability using SHAP and LIME.

| Metric | Value |
|--------|-------|
| Dataset | 205 cars, 26 original features |
| Best Model | Extra Trees Regressor (PyCaret AutoML) |
| Best R² Score | **0.9647** (96.47% variance explained) |
| Best RMSE | **$1,668** |
| Best MAE | **$968** average error per car |
| Models Trained | 21 total (3 manual + 18 AutoML) |

---

## 🗂️ Repository Structure

```
car-price-prediction/
│
├── notebooks/
│   ├── 01_Data_Gathering.ipynb          # Load dataset, inspect schema
│   ├── 02_Data_Cleaning.ipynb           # Handle nulls, types, duplicates
│   ├── 03_EDA.ipynb                     # Exploratory Data Analysis (20+ plots)
│   ├── 04_Feature_Engineering.ipynb     # 16 new features + 3 interactions
│   ├── 05_Preprocessing.ipynb           # Encoding, scaling, train/test split
│   ├── 06_Model_Training_Manual.ipynb   # Linear Regression, Decision Tree, Random Forest
│   ├── 07_PyCaret_AutoML_SHAP_LIME.ipynb # AutoML + SHAP + LIME explainability
│   └── 08_Presentation_Visualizations.ipynb # Charts for final presentation
│
├── data/
│   └── CarPrice_Assignment.csv          # Source dataset (205 cars, 26 features)
│
├── presentation/
│   └── CarPrice_Final_Presentation.pptx # Final slides
│
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## 🔬 Pipeline Overview

```
Raw Data (CSV)
     │
     ▼
01 Data Gathering ──► Load from Databricks catalog / CSV
     │
     ▼
02 Data Cleaning ───► Fix types, remove outliers (IQR), validate
     │
     ▼
03 EDA ─────────────► 20+ visualisations, correlation analysis
     │
     ▼
04 Feature Engineering ► 13 derived features + 3 interaction terms
     │
     ▼
05 Preprocessing ───► Label encoding, scaling, 80/20 split
     │
     ▼
06 Manual Models ───► Linear Regression · Decision Tree · Random Forest
     │
     ▼
07 PyCaret AutoML ──► 18 models compared → Extra Trees wins
     │
     ▼
SHAP + LIME ────────► Global + Local explainability
     │
     ▼
Production-Ready Model (96.47% R²  ·  $968 MAE  ·  <1ms/prediction)
```

---

## 📊 Model Results

### Manual Models

| Model | RMSE | R² | MAE | Notes |
|-------|------|-----|-----|-------|
| Linear Regression | $6,976 | 0.3835 | $4,975 | Baseline — severe overfitting (Train R²=1.00) |
| Decision Tree | $2,606 | 0.9140 | $1,676 | Captures non-linear patterns, depth=7 |
| Random Forest | $1,851 | 0.9566 | $1,205 | Best manual model, 100 trees |

### PyCaret AutoML — Top 3

| Rank | Model | RMSE | R² | MAE |
|------|-------|------|-----|-----|
| 🥇 1 | **Extra Trees Regressor** | **$1,668** | **0.9647** | **$968** |
| 🥈 2 | Random Forest | $1,851 | 0.9566 | $1,205 |
| 🥉 3 | Gradient Boosting | $1,920 | 0.9550 | $1,320 |

> Extra Trees uses additional randomisation at each split, reducing overfitting while maintaining high accuracy. 100 trees, max depth 15, training time ~2 seconds, prediction speed <1ms per car.

---

## 🔧 Feature Engineering

**16 new features** created from the original 26:

### Derived Features (13)
| Feature | Formula / Description |
|---------|----------------------|
| `power_to_weight` | Horsepower ÷ Curb Weight |
| `engine_efficiency` | Horsepower ÷ Engine Size |
| `avg_mpg` | (citympg + highwaympg) ÷ 2 |
| `car_volume` | Length × Width × Height |
| `wheelbase_ratio` | Wheelbase ÷ Car Length |
| `bore_stroke_ratio` | Bore ÷ Stroke |
| `price_per_hp` | Estimated Price ÷ Horsepower |
| `is_luxury` | Binary: 1 if price > 75th percentile |
| `is_high_performance` | Binary: 1 if HP > 75th percentile |
| `car_age_proxy` | Derived from symboling rating |
| `cylindernumber_numeric` | Text → numeric cylinder count |
| `avg_mpg` | Average of city and highway MPG |
| `displacement_per_cyl` | Engine Size ÷ Cylinder Count |

### Interaction Features (3)
| Feature | Formula |
|---------|---------|
| `engine_power_interaction` | `enginesize × horsepower` ← **#1 SHAP feature** |
| `size_weight_interaction` | `carlength × curbweight` |
| `efficiency_size_interaction` | `citympg × enginesize` |

**Impact of Feature Engineering:**

| Metric | Before FE | After FE | Change |
|--------|-----------|----------|--------|
| RMSE | ~$2,100 | $1,851 | **−11.9%** |
| R² | 0.9450 | 0.9566 | **+1.2%** |
| MAE | ~$1,400 | $1,205 | **−13.9%** |

---

## 🔍 Explainability — SHAP & LIME

### SHAP Global Feature Importance (Top 4)

| Rank | Feature | Avg Impact | Direction |
|------|---------|------------|-----------|
| 🥇 1 | `engine_power_interaction` | ~$2,000 | ↑ High values push price up $5K–$10K+ |
| 🥈 2 | `is_luxury` | ~$1,900 | ↑ Luxury flag adds $8K–$15K premium |
| 🥉 3 | `curbweight` | ~$1,250 | ↑ Every 1,000 lbs ≈ +$3K–$5K |
| 4 | `enginesize` | ~$1,230 | ↑ Larger displacement = higher price |

### SHAP Local Example — Sample Car

> Actual price: **$30,760** | Predicted: **$30,527** | Error: **$233 (0.76%)**

| Feature | Value | SHAP Contribution |
|---------|-------|------------------|
| engine_power_interaction | 6,120 | +$2,007 |
| curbweight | 1,989 | +$1,194 |
| enginesize | 90 | +$1,129 |
| horsepower | 68 | +$730 |
| highwaympg | 38 | −$328 |
| Base value (avg car) | — | $13,649 |

### SHAP vs LIME

| Aspect | SHAP | LIME |
|--------|------|------|
| Scope | Global + Local | Local only |
| Method | Shapley game theory | Local linear model |
| Consistency | Mathematically exact | Approximate |
| Best for | Feature importance reports | Explaining individual predictions to clients |

---

## ⚙️ Technical Stack

| Component | Tool |
|-----------|------|
| Platform | Databricks (AWS Serverless) |
| Language | Python 3.10, SQL |
| ML Framework | scikit-learn, PyCaret |
| Explainability | SHAP, LIME |
| Data Processing | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| Experiment Tracking | MLflow (Databricks-native) |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/car-price-prediction.git
cd car-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Notebooks in Order

Run each notebook sequentially. Each notebook reads from the previous output:

```
01_Data_Gathering → 02_Data_Cleaning → 03_EDA → 04_Feature_Engineering
→ 05_Preprocessing → 06_Model_Training_Manual → 07_PyCaret_AutoML_SHAP_LIME
```

### 4. On Databricks

If running on Databricks, upload `CarPrice_Assignment.csv` to the catalog first:

```python
# In notebook 01 — load from Databricks catalog
df = spark.table("workspace.default.car_price_assignment").toPandas()
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
pycaret[full]>=3.0.0
shap>=0.41.0
lime>=0.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
mlflow>=2.0.0
```

Install all with:

```bash
pip install -r requirements.txt
```

> **Note:** PyCaret requires Python 3.8–3.10. If running in Google Colab, use `!pip install pycaret lime shap --quiet` at the top of your notebook.

---

## 📈 Key Business Insights

Based on SHAP analysis, the four main price drivers are:

- **40% — Power & Performance**: `engine_power_interaction` (enginesize × horsepower) dominates all other features. Performance cars command a 50–100% price premium.
- **30% — Luxury Segment**: The `is_luxury` flag adds $8,000–$15,000 to any prediction. Brand positioning is the second most powerful lever.
- **15% — Efficiency Trade-offs**: Fuel economy (`avg_mpg`, `citympg`) has a negative correlation with price — performance buyers sacrifice MPG for power.
- **15% — Physical Characteristics**: Size, wheelbase, and curb weight correlate with pricing tier and premium materials.

### Recommendations for a Car Dealership

1. **Deploy Extra Trees** for automated pricing — ±$1,668 accuracy at <1ms per prediction
2. **Use LIME explanations** when discussing prices with customers — transparent AI builds trust and speeds the sales cycle
3. **Stock mid-tier inventory** — demand patterns are clearest here, and luxury markup is predictable
4. **Retrain quarterly** — monitor RMSE monthly and retrain immediately if degradation exceeds 15%

---

## 📝 Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| `car_ID` | int | Unique identifier |
| `symboling` | int | Insurance risk rating (−3 to +3) |
| `CarName` | str | Make and model name |
| `fueltype` | str | gas / diesel |
| `aspiration` | str | std / turbo |
| `doornumber` | str | two / four |
| `carbody` | str | sedan / hatchback / wagon / hardtop / convertible |
| `drivewheel` | str | fwd / rwd / 4wd |
| `enginelocation` | str | front / rear |
| `enginesize` | int | Displacement in cubic inches |
| `horsepower` | int | Engine power output |
| `curbweight` | int | Vehicle weight in pounds |
| `citympg` | int | City fuel economy |
| `highwaympg` | int | Highway fuel economy |
| `price` | float | **Target variable** — sale price in USD |

---

## 🤝 Acknowledgements

- Dataset sourced from course materials (Car Price Regression Assignment)
- Built as part of an AI Tools for ML Productivity course using Databricks
- AutoML powered by [PyCaret](https://pycaret.org/)
- Explainability powered by [SHAP](https://shap.readthedocs.io/) and [LIME](https://github.com/marcotcr/lime)

---

## 📄 License

This project is for educational purposes. Dataset and notebooks are shared for learning and reproducibility.

---

<div align="center">

**Built by Shakirah** · Car Price Prediction Pipeline · April 2026

*"From 38% to 96.47% accuracy — with full explainability at every step."*

</div>

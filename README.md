# Machine Learning

hands-on ML implementations covering anomaly detection, hyperparameter optimization, and model explainability.

## Isolation Forest for Air Quality Anomaly Detection

anomaly detection on air quality sensor data (4100 readings) using Isolation Forest with Optuna hyperparameter tuning and SHAP explainability.

### what it does

- detects anomalous air quality readings (PM2.5, PM10, CO2, temperature, humidity, pressure)
- tunes Isolation Forest hyperparameters using Optuna (100 trials, silhouette score optimization)
- explains model decisions using SHAP (TreeExplainer)
- assigns severity levels (High/Medium/Low) based on anomaly scores
- visualizes anomalies across different contamination thresholds

### key results

- **82 anomalies detected** out of 4100 readings (2% contamination)
- PM2.5 and PM10 are the dominant anomaly drivers (2989% and 2675% deviation from normal)
- best silhouette score: **0.7473** after Optuna optimization
- Optuna found optimal params: 78 estimators, 2.16% contamination, 0.71 max features

### techniques used

| technique | purpose |
|-----------|---------|
| Isolation Forest | unsupervised anomaly detection |
| Optuna | bayesian hyperparameter optimization (100 trials) |
| SHAP (TreeExplainer) | feature importance and model explainability |
| StandardScaler | feature normalization |
| silhouette score | cluster quality evaluation |

### visualizations

- PM2.5 time series with anomaly overlay
- anomaly score distribution over time
- contamination threshold comparison (1%, 2%, 5%)
- SHAP summary plot for feature contributions

### tech stack

- Python, pandas, NumPy, scikit-learn, matplotlib, Optuna, SHAP

### files

```
IsolationForest/
  dataset.csv                         # air quality sensor data (4100 rows, 7 features)
  isoltation-forest.ipynb             # main notebook: model + optuna + shap
  visualization-of-received-data.ipynb # EDA and data visualization
```

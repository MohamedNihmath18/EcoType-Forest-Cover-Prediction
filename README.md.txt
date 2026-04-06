# 🌲 EcoType — Forest Cover Type Prediction

A Machine Learning project that predicts forest cover type 
based on cartographic variables.

## 🎯 Project Overview
- **Domain:** Environmental Data & Geospatial Predictive Modeling
- **Model:** Random Forest Classifier
- **Accuracy:** 99.63%
- **Dataset:** UCI Forest Cover Type (145,891 rows × 13 columns)
- **Target:** 7 Forest Cover Types

## 📓 Notebooks
| Notebook | Description |
|---|---|
| 01_data_understanding | Data loading and exploration |
| 02_data_cleaning | Outlier handling and skewness |
| 03_eda_visualization | Charts and correlation analysis |
| 04_feature_engineering | Encoding and new features |
| 05_model_building | Training 5 ML models |
| 06_hyperparameter_tuning | RandomizedSearchCV tuning |
| 07_final_model_saving | Final evaluation and saving |

## 🤖 Models Compared
| Model | Accuracy |
|---|---|
| Random Forest | 99.63% ✅ |
| Decision Tree | 99.46% |
| KNN | 99.13% |
| XGBoost | 98.05% |
| Logistic Regression | 61.50% |

## 🚀 How to Run
```bash
cd app
streamlit run app.py
```

## 📦 Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn 
pip install xgboost imbalanced-learn streamlit joblib
```
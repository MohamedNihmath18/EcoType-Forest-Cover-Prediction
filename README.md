# 🌲 EcoType — Forest Cover Type Prediction

A Machine Learning project that predicts forest cover type 
based on cartographic variables.

## 🚀 Live Demo
👉 **[Click Here to Open the App](https://ecotype-forest-cover-prediction-q8nnzjzcesr2vxwarhahmv.streamlit.app/)**

## 🎯 Project Overview
- **Domain:** Environmental Data & Geospatial Predictive Modeling
- **Best Model:** Random Forest Classifier
- **Best Accuracy:** 99.63%
- **Deployed Model:** XGBoost (98.05%)
- **Dataset:** UCI Forest Cover Type (145,891 rows × 13 columns)
- **Target:** 7 Forest Cover Types

## 🌿 Forest Cover Types
| Class | Forest Type |
|---|---|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

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
| Random Forest | 99.63% ✅ Best |
| Decision Tree | 99.46% |
| KNN | 99.13% |
| XGBoost | 98.05% 🚀 Deployed |
| Logistic Regression | 61.50% |

## 📊 Key Findings
- **Class Imbalance:** Lodgepole Pine = 70.6% of dataset
- **Top Feature:** Elevation (most important predictor)
- **Strongest Correlation:** Hillshade_9am ↔ Hillshade_3pm (-0.81)
- **Imbalance Fix:** RandomOverSampler used to balance all 7 classes

## 📁 Project Structure

EcoType_Project/
│
├── data/
│   └── forest_cover.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda_visualization.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_building.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   └── 07_final_model_saving.ipynb
│
├── models/
│   ├── xgboost.pkl
│   └── le_target.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md

## ⚠️ Large Files Note
Due to GitHub file size limits, these files are not included:
- `models/best_model.pkl` (195MB)
- `models/random_forest.pkl` (195MB)
- `models/knn.pkl` (74MB)
- `data/` CSV files

To reproduce — download dataset from:
👉 https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset

Then run all notebooks in order (01 to 07)

## 🚀 How to Run Locally
```bash
# Clone the repository
git clone https://github.com/MohamedNihmath18/EcoType-Forest-Cover-Prediction.git

# Navigate to app folder
cd EcoType-Forest-Cover-Prediction/app

# Run the app
streamlit run app.py
```

## 📦 Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn 
pip install xgboost imbalanced-learn streamlit joblib
```

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-green)

## 👨‍💻 Author
**Mohamed Nihmath**
- GitHub: [@MohamedNihmath18](https://github.com/MohamedNihmath18)
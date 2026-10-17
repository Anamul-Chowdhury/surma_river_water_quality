# Surma River Water Quality Classification

Paper: https://ieeexplore.ieee.org/document/11491222/

Reproduces the methodology of the ICCIT 2025 paper "Machine Learning Approaches to
Classify and Predict Water Quality Status in the Surma River, Sylhet, Bangladesh".

## Dataset

`surma_water_quality.csv` in the same folder. Columns needed:
pH, DO, BOD, COD, TDS, SS, EC, Temperature, WQI_Status (values: Bad, Marginal, Fair).

## What each cell does

1. Preprocessing: median fill then KNN(k=5) imputation, duplicate removal, z-score
   standardization, stratified 80/20 train/test split.
2. Training: Logistic Regression (C=1.0), KNN (k=5), SVM (RBF, C=10),
   Random Forest (200 trees, depth 12), XGBoost (200 trees, depth 6, lr 0.1),
   LightGBM (200 trees, depth 8, lr 0.1). Outputs accuracy, precision, recall, F1,
   ROC-AUC per model and saves `surma_model_results.csv`.
3. 5-fold stratified cross validation, saves `surma_cv_results.csv`.
4. Analysis of the best model: class-wise report, confusion matrix
   (`surma_confusion_matrix.png`), ROC curves (`surma_roc.png`), SHAP summary
   (`surma_shap.png`).

## Requirements

pandas, numpy, scikit-learn, xgboost, lightgbm, shap, matplotlib, seaborn.

## Expected outcome

LightGBM should be the top model, matching the paper (accuracy ~95.52%).

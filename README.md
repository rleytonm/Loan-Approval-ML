# Loan-Approval-ML
Loan Approval and Maximum Loan Amount Prediction
Machine Learning for Consumer Credit Risk

# Project Overview

This project develops machine learning models to support lending decisions in a retail banking context. It consists of:

Part A – Loan Approval Classification: Predict whether a loan application should be approved or rejected.

Part B – Maximum Loan Amount Prediction: Estimate the maximum loan amount a client should be granted.

The modelling approach combines academic justification with business-oriented evaluation.

# Business Context

Credit decisions involve asymmetric costs:

- Approving risky clients leads to financial losses.

- Rejecting safe clients results in opportunity cost.

- Overestimating loan amounts increases default risk.

Models were evaluated not only statistically, but also through a cost-based business simulation.

# Data Preparation

- Feature selection supported by academic literature
- Removal of variables with severe missingness or leakage risk
- Outlier treatment and categorical encoding
- 80/20 train–test split with fixed random state for reproducibility

# Part A – Loan Approval Classification

Models compared:

- Logistic Regression
- Random Forest
- Naïve Bayes

Evaluation prioritised Recall, Precision, and F1-Score to reflect business objectives.

Selected model: Random Forest, offering the best balance between detecting risky clients and limiting false positives.

Hyperparameter tuning with 5-fold cross-validation improved overall simulated profitability, although some misclassifications remain a concern for real-world deployment.

# Part B – Maximum Loan Amount Prediction

Decision Tree regression models were developed due to their interpretability and robustness without feature scaling.

Evaluation metrics:

- MSE
- MAE
- R²

The selected model achieved very high explanatory power (R² ≈ 1.0) with low average prediction error. Pruning reduced performance significantly.

# Key Contributions

- Alignment between ML metrics and business impact
- Explicit modelling of asymmetric financial costs
- Consideration of data leakage and multicollinearity
- Ethical reflection on automated credit decisions

Tech Stack

Python, Scikit-learn, PyCaret, Pandas, NumPy, Matplotlib

Author:
Cristian Renato Leyton Medina
MSc Data Science and Analytics
University of Westminster

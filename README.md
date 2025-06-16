# Linear Regression from Scratch

Creating a multivariate linear regression model from scratch in Python — without using the sklearn library.

## Overview

This project demonstrates how to build, train, and evaluate a linear regression model using only core Python libraries and popular data science packages (Pandas, NumPy, Matplotlib, Seaborn). The model is developed from the ground up, including the full implementation of gradient descent, data standardization, and model metrics. The project uses the [Student Performance (Multiple Linear Regression) dataset](https://www.kaggle.com/datasets/nikhilsinghrao/student-performance-multiple-linear-regression) from Kaggle, containing 10,000 student records.

## Features

- **Manual Data Preprocessing:** Feature engineering, conversion of categorical to numerical data, and standardization.
- **Exploratory Data Analysis:** Visualizations and statistical summaries to understand data distributions and feature importance.
- **Custom Train/Test Split:** Function to manually split data (instead of using sklearn).
- **Linear Regression Implementation:** Model class (with fit and predict methods) using gradient descent for optimization.
- **Performance Metrics:** Tracks and visualizes R², MSE, and MAE across training epochs.
- **Visualization:** Includes feature distributions, bivariate relationships, and correlation matrix.

## Key Implementation Details

- **No sklearn:** All data splitting and modeling is implemented manually.
- **Gradient Descent:** The model updates weights and bias using gradient descent, tracking R², MSE, and MAE during training.
- **Feature Analysis:** Includes EDA with histograms, violin plots, hexbin plots, and a correlation heatmap to evaluate feature relationships.

## Results Summary

- **Strongest Predictor:** 'Previous Scores' (Pearson correlation ≈ 0.92)
- **Other Predictors:** 'Hours Studied' and 'Sample Question Papers Practiced' show moderate correlation; 'Sleep Hours' and 'Extracurricular Activities' are weak predictors.
- **Model Performance:** Performance metrics (R², MSE, MAE) are tracked and visualized through training.

## Motivation

This project aims to deepen understanding of the mathematics and mechanics behind linear regression, providing a foundation for further studies in machine learning by removing reliance on high-level libraries.

## Acknowledgements

- Dataset: [Student Performance (Multiple Linear Regression) by Nikhil Narayan](https://www.kaggle.com/datasets/nikhilsinghrao/student-performance-multiple-linear-regression)
- Inspired by open-source data science learning resources.

## License

MIT License

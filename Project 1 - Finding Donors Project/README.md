# Project 1: Finding Donors for CharityML

This repository contains **Project 1** from the *Intro to Machine Learning with TensorFlow Nanodegree*. The goal of this project is to apply supervised learning algorithms to U.S. Census data in order to identify individuals most likely to donate to CharityML, a fictitious nonprofit organization.

## Overview

The project evaluates the performance of different supervised machine learning algorithms on structured census data. It involves data preprocessing, algorithm benchmarking, model optimization, and final evaluation using accuracy and F-score. The best-performing model is then analyzed to understand key predictive features.

## Main Features

* **Data Preprocessing**: Cleans and transforms the raw dataset, including encoding categorical variables and scaling numeric features.
* **Model Benchmarking**: Tests the performance of three different classifiers on multiple training set sizes.
* **Model Optimization**: Applies grid search to fine-tune hyperparameters for the selected model.
* **Performance Evaluation**: Compares models using accuracy, F-score, training time, and prediction time.
* **Feature Analysis**: Ranks the top 5 features contributing to the final model’s decision-making.

## Functional Workflow

1. **Exploratory Analysis**:

   * Examine income distribution and class imbalance
   * Log-transform skewed features such as `capital-gain` and `capital-loss`

2. **Preprocessing**:

   * Encode categorical variables using one-hot encoding
   * Normalize numerical features
   * Split data into training and testing sets

3. **Model Evaluation**:

   * Test three classifiers (e.g., Decision Tree, SVM, AdaBoost)
   * Train each model on three subsets (1%, 10%, 100%) of the data
   * Evaluate with F-score and accuracy

4. **Model Tuning**:

   * Optimize the best model using `GridSearchCV`
   * Evaluate optimized model on the full test set

5. **Interpretation**:

   * Extract and visualize top 5 most predictive features
   * Analyze model predictions for fairness and effectiveness

## Tools and Technologies

* Python 3.x
* scikit-learn
* pandas
* numpy
* matplotlib
* Jupyter Notebook

## Output Metrics

* Accuracy and F-score for each model and training set size
* Best-performing model before and after tuning
* Feature importance ranking for top predictors
* Visualizations comparing model performance across metrics

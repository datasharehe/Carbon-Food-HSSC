# Data and Code

This repository provides the data and Python code supporting the study:

**The coordinated mechanism of food security and agricultural carbon reduction and sink based on interpretable machine learning models**

## Data

The data cover 31 provincial-level regions in China from 2001 to 2022. The accompanying data files provide the observations used in the analysis and explanations of the variables.

## Code

The Python code includes model training, hyperparameter tuning, performance evaluation, and visualization. Five regression models are considered:

- Decision Tree
- Random Forest
- Support Vector Regression
- XGBoost
- LightGBM

Samples are divided chronologically into training and test sets at an approximate ratio of 8:2, keeping observations from the same year together. Year and province dummy variables are included. Hyperparameter tuning uses five-fold cross-validation grouped by year. XGBoost uses early stopping after 50 consecutive rounds without improvement.

## Requirements

The code uses the following Python packages:

- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- shap
- matplotlib
- seaborn

## Usage

Install the required packages, download the data and code, and update the file paths in the script to match your local directory before running the analysis.

## Citation

If you use these data or code, please cite the associated paper. Full publication details will be added when available.

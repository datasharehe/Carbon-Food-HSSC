
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


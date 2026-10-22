
# Set data and output paths.
BASE_DIR = r"D:\博士相关\202608"
DATA_FILE = os.path.join(BASE_DIR, "regdata.csv")
IMG_DIR = os.path.join(BASE_DIR, "Fig.3")
os.makedirs(IMG_DIR, exist_ok=True)

# Calculate regression metrics and append them to a CSV file.
def performance_reg(model, X, y, metrics_name=None):
    y_pred = model.predict(X)
    metrics = {
        "Mean Squared Error (MSE)": mean_squared_error(y, y_pred),
        "Root Mean Squared Error (RMSE)": np.sqrt(mean_squared_error(y, y_pred)),
        "Mean Absolute Error (MAE)": mean_absolute_error(y, y_pred),
        "R2 Score": r2_score(y, y_pred)
    }
    if metrics_name:
        print(metrics_name, ":")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(os.path.join(IMG_DIR, 'Fit.csv'), mode='a', header=False, index=False)

# Load the dataset.
data = pd.read_csv(DATA_FILE)

# Remove incomplete observations and select predictors and target.
data = data.dropna()
X = data[['PSA', 'DA', 'TEC', 'LN', 'EY', 'FU', 'EI', 'ME', 'UY', 'GDP', 'GDP2', 'UR', 'AE', 'PG', 'IS']]
y = data['CCD']

# Identify the panel identifier columns.
def identify_panel_column(candidates, meaning):
    matches = [column for column in data.columns
               if str(column).strip().lower() in candidates]
    if len(matches) != 1:
        raise ValueError(f'{meaning}列匹配结果：{matches}')
    return matches[0]

year_column = identify_panel_column({'year'}, 'year')
province_column = identify_panel_column({'province'}, 'province')
year_values = pd.to_numeric(data[year_column], errors='raise')
years = np.sort(year_values.unique())

# Split complete years chronologically into approximately 80% training and 20% testing.
split_postion = int(np.floor(len(years) * 0.8 + 0.5))
train_mask = year_values.isin(years[:split_position])
test_mask = ~train_mask

# Add year and province dummy variables, aligned to training-period columns.
fixed_effects = pd.get_dummies(
    data[[year_column, province_column]].astype(str),
    columns=[year_column, province_column], dtype=float)
training_fixed_effects = pd.get_dummies(
    data.loc[train_mask, [year_column, province_column]].astype(str),
    columns=[year_column, province_column], dtype=float)
fixed_effects = fixed_effects.reindex(columns=training_fixed_effects.columns, fill_value=0)
X = pd.concat([X, fixed_effects], axis=1)
X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]
training_years = year_values.loc[train_mask]
# Keep observations from the same year together in five-fold cross-validation.
cv = GroupKFold(n_splits=5)

# Define the hyperparameter search grids.
param_grids = {
    'Decision Tree': {'max_depth': [5, 10], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 5]},
    'Random Forest': {'n_estimators': [100, 300], 'max_depth': [None, 10],
                      'min_samples_leaf': [1, 5]},
    'SVR': {'C': [1.0, 10.0], 'epsilon': [0.01, 0.1], 'gamma': ['scale', 'auto']},
    'XGBoost': {'max_depth': [3, 6], 'learning_rate': [0.05, 0.1],
                'reg_lambda': [1.0, 10.0]},
    'LightGBM': {'n_estimators': [100, 300], 'num_leaves': [15, 31],
                 'learning_rate': [0.05, 0.1]}
}

# Select hyperparameters using cross-validation grouped by year.
def tune_model(model, model_name):
    search = GridSearchCV(model, param_grids[model_name], cv=cv,
                          scoring='neg_mean_squared_error', refit=True)
    search.fit(X_train, y_train, groups=training_years)
    print(model_name, 'best parameters:', search.best_params_)
    return search.best_estimator_

dt_regressor = tune_model(DecisionTreeRegressor(random_state=2021), 'Decision Tree')
rf_regressor = tune_model(RandomForestRegressor(random_state=2021), 'Random Forest')
svr_regressor = tune_model(SVR(), 'SVR')

# Tune XGBoost with fold-specific validation and early stopping after 50 rounds without improvement.
xgb_cv_results = []
for parameters in ParameterGrid(param_grids['XGBoost']):
    fold_losses = []
    for fit_indices, validation_indices in cv.split(X_train, y_train, training_years):
        fold_model = xgb.XGBRegressor(
            objective='reg:squarederror', random_state=2021,
            n_estimators=2000, eval_metric='rmse', early_stopping_rounds=50,
            **parameters)
        fold_model.fit(
            X_train.iloc[fit_indices], y_train.iloc[fit_indices],
            eval_set=[(X_train.iloc[validation_indices], y_train.iloc[validation_indices])],
            verbose=False)
        fold_losses.append(mean_squared_error(
            y_train.iloc[validation_indices],
            fold_model.predict(X_train.iloc[validation_indices])))
    xgb_cv_results.append((np.mean(fold_losses), parameters))

best_parameters = min(xgb_cv_results, key=lambda result: result[0])[1]

# Use the latest training year to select the number of boosting rounds.
early_validation_mask = training_years == training_years.max()
xgb_regressor = xgb.XGBRegressor(
    objective='reg:squarederror', random_state=2021,
    n_estimators=2000, eval_metric='rmse', early_stopping_rounds=50,
    **best_parameters)
xgb_regressor.fit(
    X_train.loc[~early_validation_mask], y_train.loc[~early_validation_mask],
    eval_set=[(X_train.loc[early_validation_mask], y_train.loc[early_validation_mask])],
    verbose=False)
# Refit XGBoost on all training observations using the selected number of rounds.
best_rounds = xgb_regressor.best_iteration + 1
xgb_regressor = xgb.XGBRegressor(
    objective='reg:squarederror', random_state=2021,
    n_estimators=best_rounds, eval_metric='rmse', **best_parameters)
xgb_regressor.fit(X_train, y_train)
print('XGBoost best parameters:', best_parameters, 'n_estimators:', best_rounds)

lgb_regressor = tune_model(lgb.LGBMRegressor(random_state=2021), 'LightGBM')
# Report training and test performance for each model.
print("-----Decision Tree-----")
performance_reg(dt_regressor, X_train, y_train, metrics_name='train')
performance_reg(dt_regressor, X_test, y_test, metrics_name='test')

print("-----Random Forest-----")
performance_reg(rf_regressor, X_train, y_train, metrics_name='train')
performance_reg(rf_regressor, X_test, y_test, metrics_name='test')

print("-----SVR-----")
performance_reg(svr_regressor, X_train, y_train, metrics_name='train')
performance_reg(svr_regressor, X_test, y_test, metrics_name='test')

print("-----XGBoost-----")
performance_reg(xgb_regressor, X_train, y_train, metrics_name='train')
performance_reg(xgb_regressor, X_test, y_test, metrics_name='test')

print("-----LightGBM-----")
performance_reg(lgb_regressor, X_train, y_train, metrics_name='train')
performance_reg(lgb_regressor, X_test, y_test, metrics_name='test')

# Plot observed versus predicted values for each model.
models = [dt_regressor, rf_regressor, svr_regressor, xgb_regressor, lgb_regressor]
model_names = ['Decision Tree', 'Random Forest', 'SVR', 'XGBoost', 'LightGBM']
for i, model in enumerate(models):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    r2_train_str = f"{r2_train:.5f}"[:str(r2_train).find('.')+4]
    r2_test_str = f"{r2_test:.5f}"[:str(r2_test).find('.')+4]

    data_train = pd.DataFrame({
        'True': y_train,
        'Predicted': y_pred_train,
        'Data Set': 'Train',
        'Model': model_names[i]
    })

    data_test = pd.DataFrame({
        'True': y_test,
        'Predicted': y_pred_test,
        'Data Set': 'Test',
        'Model': model_names[i]
    })

    data = pd.concat([data_train, data_test])

    plt.figure(figsize=(8, 6), dpi=1200)
    g = sns.JointGrid(data=data, x="True", y="Predicted", hue="Model", height=10, space=0)

    g.plot_joint(sns.scatterplot, alpha=0.5)
    sns.regplot(data=data_train, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#b4d4e1',
                label='Train Regression Line')
    sns.regplot(data=data_test, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#f4ba8a',
                label='Test Regression Line')

    ax = g.ax_joint
    ax.text(0.6, 0.97, f'Model = {model_names[i]}', transform=ax.transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    ax.text(0.95, 0.11, f'Train $R^2$ = {r2_train_str}', transform=ax.transAxes, fontsize=18,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    ax.text(0.95, 0.05, f'Test $R^2$ = {r2_test_str}', transform=ax.transAxes, fontsize=18,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    ax.set_xlabel('True Values', fontsize=20, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18)

    ax.plot([data['True'].min(), data['True'].max()], [data['True'].min(), data['True'].max()], c="black", alpha=0.5,
            linestyle='--', label='x=y')
    ax.legend(fontsize=18, title_fontsize='19')

    
    # Save the model fit plot.
    plt.savefig(os.path.join(IMG_DIR, f"{model_names[i]}_TrueFalse.png"),
                format='png', bbox_inches='tight', dpi=300)
    
    plt.show()

    # Plot SHAP dependence values colored by the paired feature.
    def plot_two_dimensional_shap():
             pairs = [('FU', 'EY'), ('PSA', 'GDP')]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
        export_frames = []

        for i, (ax, (focal, color_feature)) in enumerate(zip(axes, pairs)):
            focal_index = feature_names.index(focal)
            color_values = X[color_feature].to_numpy()
            scatter = ax.scatter(
                X[focal].to_numpy(),
                shap_values[:, focal_index],
                c=color_values,
                cmap='coolwarm',
                s=24,
                alpha=0.78,
                edgecolors='none'
            )
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.set_xlabel(focal, fontsize=15, fontweight='bold')
            ax.set_ylabel(f'SHAP value for {focal}', fontsize=15)
            ax.tick_params(axis='both', labelsize=13)
            ax.text(0.03, 0.95, f'({chr(97 + i)})', transform=ax.transAxes,
                    fontsize=20, ha='left', va='top')
            cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label(color_feature, fontsize=15)
            cbar.ax.tick_params(labelsize=12)

            export_frames.append(pd.DataFrame({
                'Observation': np.arange(len(X)) + 1,
                'Panel': chr(97 + i),
                'Focal_feature': focal,
                'Focal_value': X[focal].to_numpy(),
                'SHAP_value': shap_values[:, focal_index],
                'Color_feature': color_feature,
                'Color_feature_value': color_values
            }))

        # Export the SHAP figures and their underlying plotting data.
        fig.tight_layout(w_pad=2.2)
        fig.savefig(os.path.join(IMG_DIR, 'Fig9_Two_Dimensional_SHAP_Dependence.png'),
                    dpi=600, bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(IMG_DIR, 'Fig9_Two_Dimensional_SHAP_Dependence.pdf'),
                    bbox_inches='tight', facecolor='white')
        pd.concat(export_frames, ignore_index=True).to_csv(
            os.path.join(IMG_DIR, 'Fig9_Two_Dimensional_SHAP_Dependence_Data.csv'),
            index=False, encoding='utf-8-sig'
        )
        plt.show()
        plt.close(fig)

    plot_two_dimensional_shap()

    print('end', IMG_DIR)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import boxcox, shapiro

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from joblib import dump

import warnings
warnings.filterwarnings("ignore")


def linear_regression_best_model(x_train_scaled, x_test_scaled, y_train):

    model = LinearRegression()
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)

    return y_pred, model


def random_forest_best_model(x_train, x_test, y_train):

    model = RandomForestRegressor(random_state=2706)

    param_grid = {
        'n_estimators': [25, 50, 100, 200, 250],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(x_test)

    return y_pred, best_model


def svr_best_model(x_train_scaled, x_test_scaled, y_train):

    model = SVR(kernel='linear')

    param_grid = {
        'C': [0.1, 0.5, 1, 5, 10],
        'epsilon': [0.1, 0.2, 0.5],
        'kernel': ['linear', 'rbf', 'poly']
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(x_train_scaled, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(x_test_scaled)

    return y_pred, best_model


def gradient_boost_best_model(x_train, x_test, y_train):

    model = GradientBoostingRegressor(random_state=2706)

    param_grid = {
        'n_estimators': [25, 50, 100, 200, 250],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(x_test)

    return y_pred, best_model


def model_performance_analysis(model, model_name, y_pred, y_test, fitted_lambda):

    pred_vs_test = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
    pred_vs_test['Error'] = round(pred_vs_test['Actual'] - pred_vs_test['Predicted'], 2)

    n_rmse_mean = np.sqrt(np.mean(pred_vs_test['Error']**2)) / y_test.mean()

    n_rmse_range = np.sqrt(np.mean(pred_vs_test['Error']**2)) / (y_test.max() - y_test.min())

    r2 = r2_score(y_test, y_pred)

    models_metrics.append({'model': model_name,
                           'model_object': model,
                           'n_rmse_mean': n_rmse_mean,
                           'n_rmse_range': n_rmse_range,
                           'r2': r2})


df = pd.read_csv("data/gold_layer/casks_database.csv")

df = df[df['strength'] >= 40]

df = df[['age', 'distillery', 'region', 'cask_type', 'cask_filling', 'inf_adj_hammer_price_per_litre_of_alcohol']]

categorical_columns = ['distillery', 'region', 'cask_type', 'cask_filling']
df[categorical_columns] = df[categorical_columns].fillna('Undisclosed')

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df['distillery'] = df['distillery'].str.lower()
df['region'] = df['region'].str.lower()
df['cask_type'] = df['cask_type'].str.lower()
df['cask_filling'] = df['cask_filling'].str.lower()

df = pd.get_dummies(df, columns=categorical_columns, drop_first=False, dtype=int)

df.columns = [col.replace(' ', '_') for col in df.columns]

df = df[['age', 'distillery_macallan', 'distillery_springbank', 'region_campbeltown', 'cask_type_hogshead',
         'cask_type_butt', 'cask_filling_second_fill', 'inf_adj_hammer_price_per_litre_of_alcohol']]

y_boxcox, fitted_lambda = boxcox(df['inf_adj_hammer_price_per_litre_of_alcohol'])

df['inf_adj_hammer_price_per_litre_of_alcohol_boxcox'] = y_boxcox

Q1 = df['inf_adj_hammer_price_per_litre_of_alcohol_boxcox'].quantile(0.25)
Q3 = df['inf_adj_hammer_price_per_litre_of_alcohol_boxcox'].quantile(0.75)
IQR = Q3 - Q1

filtro = (df['inf_adj_hammer_price_per_litre_of_alcohol_boxcox'] >= (Q1 - 1.5 * IQR)) & (df['inf_adj_hammer_price_per_litre_of_alcohol_boxcox'] <= (Q3 + 1.5 * IQR))
df = df[filtro].reset_index(drop=True)

df = df.drop(columns=['inf_adj_hammer_price_per_litre_of_alcohol'])

x = df.drop(columns=['inf_adj_hammer_price_per_litre_of_alcohol_boxcox'])
y = df['inf_adj_hammer_price_per_litre_of_alcohol_boxcox']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

x_columns_features = x.columns.to_list()

scaler_x = StandardScaler()

x_train_scaled = x_train.copy()
x_train_scaled[['age']] = scaler_x.fit_transform(x_train_scaled[['age']])

x_test_scaled = x_test.copy()
x_test_scaled[['age']] = scaler_x.transform(x_test_scaled[['age']])

models_metrics = []
trained_models = {}

total_tests = 4
finished_tests = 0

# Linear Regression

y_pred, best_model = linear_regression_best_model(x_train_scaled, x_test_scaled, y_train)
model_performance_analysis(best_model, 'Linear Regression', y_pred, y_test, fitted_lambda)

finished_tests += 1
print("".join(finished_tests * ['|'] + (total_tests - finished_tests) * ['-']) + f' {finished_tests}/{total_tests}\n')

# Support Vector Regression

y_pred, best_model = svr_best_model(x_train_scaled, x_test_scaled, y_train)
model_performance_analysis(best_model, 'Support Vector Regression', y_pred, y_test, fitted_lambda)

finished_tests += 1
print("".join(finished_tests * ['|'] + (total_tests - finished_tests) * ['-']) + f' {finished_tests}/{total_tests}\n')

# Random Forest

y_pred, best_model = random_forest_best_model(x_train, x_test, y_train)
model_performance_analysis(best_model, 'Random Forest', y_pred, y_test, fitted_lambda)

finished_tests += 1
print("".join(finished_tests * ['|'] + (total_tests - finished_tests) * ['-']) + f' {finished_tests}/{total_tests}\n')

# Gradient Boosting

y_pred, best_model = gradient_boost_best_model(x_train, x_test, y_train)
model_performance_analysis(best_model, 'Gradient Boost', y_pred, y_test, fitted_lambda)

finished_tests += 1
print("".join(finished_tests * ['|'] + (total_tests - finished_tests) * ['-']) + f' {finished_tests}/{total_tests}\n')

df_metrics = pd.DataFrame(models_metrics)
df_metrics['n_rmse_mean'] = df_metrics['n_rmse_mean'].round(4)
df_metrics['n_rmse_range'] = df_metrics['n_rmse_range'].round(4)
df_metrics['r2'] = df_metrics['r2'].round(4)

df_metrics = df_metrics.loc[df_metrics['r2'] > 0, ]
df_metrics = df_metrics.sort_values('n_rmse_mean').reset_index(drop=True).reset_index().rename(columns={'index': 'ranking_n_rmse_mean'})
df_metrics = df_metrics.sort_values('n_rmse_range').reset_index(drop=True).reset_index().rename(columns={'index': 'ranking_n_rmse_range'})
df_metrics['overall_score'] = df_metrics['ranking_n_rmse_mean'] + df_metrics['ranking_n_rmse_range']
df_metrics = df_metrics.sort_values('overall_score').reset_index(drop=True).reset_index().rename(columns={'index': 'overall_ranking'})

print(df_metrics)

top3_models_info = df_metrics.head(3)

for i in range(3):
    print(top3_models_info.iloc[i]['model'])
    print(f"models/top{i+1}_model.pkl")
    model = top3_models_info.iloc[i]['model_object']
    dump(model, f"models/top{i+1}_model.pkl")


y_pred_ensemble = 0

for i in range(3):

    model_info = top3_models_info.iloc[i]

    if model_info['model'] in ['Linear Regression', 'Support Vector Regression', 'Artificial Neural Network']:
        X = x_test_scaled
    else:
        X = x_test

    y_pred = model_info['model_object'].predict(X)

    y_pred_ensemble += y_pred

y_pred_ensemble = y_pred_ensemble / 3

pred_vs_test = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred_ensemble})
pred_vs_test['Error'] = round(pred_vs_test['Actual'] - pred_vs_test['Predicted'], 2)

n_rmse_mean = np.sqrt(np.mean(pred_vs_test['Error']**2)) / y_test.mean()

n_rmse_range = np.sqrt(np.mean(pred_vs_test['Error']**2)) / (y_test.max() - y_test.min())

r2 = r2_score(y_test, y_pred)

df_ensemble_model = pd.DataFrame([{'model': 'Ensemble Model',
                                    'n_rmse_mean':  round(n_rmse_mean, 2),
                                    'n_rmse_range': round(n_rmse_range, 2),
                                    'r2': round(r2, 2)}])

top3_models_info = top3_models_info[['model', 'model_object', 'n_rmse_mean', 'n_rmse_range', 'r2', 'overall_ranking']]

df_ensemble_model_top3_models = pd.concat([top3_models_info, df_ensemble_model], ignore_index=True)

df_ensemble_model_top3_models = df_ensemble_model_top3_models.sort_values('n_rmse_mean').reset_index(drop=True).reset_index().rename(columns={'index': 'ranking_n_rmse_mean'})
df_ensemble_model_top3_models = df_ensemble_model_top3_models.sort_values('n_rmse_range').reset_index(drop=True).reset_index().rename(columns={'index': 'ranking_n_rmse_range'})
df_ensemble_model_top3_models['overall_score'] = df_ensemble_model_top3_models['ranking_n_rmse_mean'] + df_ensemble_model_top3_models['ranking_n_rmse_range']
df_ensemble_model_top3_models = df_ensemble_model_top3_models.sort_values('overall_score').reset_index(drop=True)

print(df_ensemble_model_top3_models)

df_ensemble_model_top3_models.to_csv('models/models_metrics/models_metrics.csv', index=False)

dump(fitted_lambda, 'models/variables_for_production_model/fitted_lambda.pkl')
dump(scaler_x, 'models/variables_for_production_model/scaler_x.pkl')
dump(x_columns_features, 'models/variables_for_production_model/x_columns_features.pkl')
import pandas as pd
import numpy as np
import os

from scipy.special import inv_boxcox

from joblib import load

from dotenv import load_dotenv

load_dotenv()

x_columns_corr_sel = os.getenv("x_columns_corr_sel").split(",")

fitted_lambda_standard = float(os.getenv("fitted_lambda_standard"))
fitted_lambda_per_rla = float(os.getenv("fitted_lambda_per_rla"))
fitted_lambda_per_rla_times_age = float(os.getenv("fitted_lambda_per_rla_times_age"))

scaler_x_standard = load('scalers/scaler_x_standard.pkl')
scaler_x_per_rla = load('scalers/scaler_x_per_rla.pkl')
scaler_x_per_rla_times_age = load('scalers/scaler_x_per_rla_times_age.pkl')

models_metrics = pd.read_csv('models_metrics/models_metrics.csv')

df_top3_models_metrics = models_metrics[models_metrics['model'] != 'Ensemble Model'].reset_index(drop=True)
df_top3_models_metrics = df_top3_models_metrics.sort_values('overall_ranking').reset_index(drop=True)

def ensemble_model_predict(x_input):

    y_pred_ensemble = 0

    for i in range(3):

        model_info = df_top3_models_metrics.iloc[i]

        model = load(f'models/top{i+1}_model.pkl')

        if (model_info['features'] == 'Correlation selected features'):
            X = x_input[x_columns_corr_sel]
        else:
            X = x_input

        if model_info['model'] in ['Linear Regression', 'Support Vector Regression']:
            if (model_info['y_variable'] == 'Inflation Adjusted Hammer Price'):
                X = scaler_x_standard.transform(X)
            elif (model_info['y_variable'] == 'Inflation Adjusted Hammer Price per Litre of Alcohol'):
                X = scaler_x_per_rla.transform(X)
            elif (model_info['y_variable'] == 'Inflation Adjusted Hammer Price per Litre of Alcohol Times Age'):
                X = scaler_x_per_rla_times_age.transform(X)

        y_pred = model.predict(X)

        if (model_info['transformation'] == 'Box-Cox transformation') and (model_info['y_variable'] == 'Inflation Adjusted Hammer Price'):
            y_pred = inv_boxcox(y_pred, fitted_lambda_standard)
        elif (model_info['transformation'] == 'Box-Cox transformation') and (model_info['y_variable'] == 'Inflation Adjusted Hammer Price per Litre of Alcohol'):
            y_pred = inv_boxcox(y_pred, fitted_lambda_per_rla)
        elif (model_info['transformation'] == 'Box-Cox transformation') and (model_info['y_variable'] == 'Inflation Adjusted Hammer Price per Litre of Alcohol Times Age'):
            y_pred = inv_boxcox(y_pred, fitted_lambda_per_rla_times_age)
        elif model_info['transformation'] == 'Log transformation':
            y_pred = np.exp(y_pred)

        if model_info['y_variable'] == 'Inflation Adjusted Hammer Price per Litre of Alcohol':
            y_pred = y_pred * x_input['rla'].values
        elif model_info['y_variable'] == 'Inflation Adjusted Hammer Price per Litre of Alcohol Times Age':
            y_pred = y_pred * x_input['rla'].values / x_input['age'].values

        y_pred_ensemble += y_pred

    y_pred_ensemble = y_pred_ensemble / 3
    y_pred_ensemble = y_pred_ensemble[0]
    y_pred_ensemble = round(y_pred_ensemble, 2)

    return y_pred_ensemble



def best_model_predict(x_input):

    model_info = df_top3_models_metrics.iloc[0]

    model = load(f'models/top1_model.pkl')

    if (model_info['features'] == 'Correlation selected features'):
        X = x_input[x_columns_corr_sel]
    else:
        X = x_input

    if model_info['model'] in ['Linear Regression', 'Support Vector Regression']:
        if (model_info['y_variable'] == 'Inflation Adjusted Hammer Price'):
            X = scaler_x_standard.transform(X)
        elif (model_info['y_variable'] == 'Inflation Adjusted Hammer Price per Litre of Alcohol'):
            X = scaler_x_per_rla.transform(X)
        elif (model_info['y_variable'] == 'Inflation Adjusted Hammer Price per Litre of Alcohol Times Age'):
            X = scaler_x_per_rla_times_age.transform(X)

    y_pred = model.predict(X)

    if (model_info['transformation'] == 'Box-Cox transformation') and (model_info['y_variable'] == 'Inflation Adjusted Hammer Price'):
        y_pred = inv_boxcox(y_pred, fitted_lambda_standard)
    elif (model_info['transformation'] == 'Box-Cox transformation') and (model_info['y_variable'] == 'Inflation Adjusted Hammer Price per Litre of Alcohol'):
        y_pred = inv_boxcox(y_pred, fitted_lambda_per_rla)
    elif (model_info['transformation'] == 'Box-Cox transformation') and (model_info['y_variable'] == 'Inflation Adjusted Hammer Price per Litre of Alcohol Times Age'):
        y_pred = inv_boxcox(y_pred, fitted_lambda_per_rla_times_age)
    elif model_info['transformation'] == 'Log transformation':
        y_pred = np.exp(y_pred)

    if model_info['y_variable'] == 'Inflation Adjusted Hammer Price per Litre of Alcohol':
        y_pred = y_pred * x_input['rla'].values
    elif model_info['y_variable'] == 'Inflation Adjusted Hammer Price per Litre of Alcohol Times Age':
        y_pred = y_pred * x_input['rla'].values / x_input['age'].values

    y_pred = round(y_pred, 2)

    return y_pred



def cask_worth_predict(x_input):

    if models_metrics.iloc[0]['model'] == 'Ensemble Model':
        return ensemble_model_predict(x_input)
    else:
        return best_model_predict(x_input)
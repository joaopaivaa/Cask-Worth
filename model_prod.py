import pandas as pd
from scipy.special import inv_boxcox
from joblib import load

fitted_lambda = load('variables_for_production_model/fitted_lambda.pkl')
scaler_x = load('variables_for_production_model/scaler_x.pkl')

models_metrics = pd.read_csv('models_metrics/models_metrics.csv')

df_top3_models_metrics = models_metrics[models_metrics['model'] != 'Ensemble Model'].reset_index(drop=True)
df_top3_models_metrics = df_top3_models_metrics.sort_values('overall_ranking').reset_index(drop=True)

def ensemble_model_predict(x_input):

    y_pred_ensemble = 0

    for i in range(3):

        model_info = df_top3_models_metrics.iloc[i]

        model = load(f'models/top{i+1}_model.pkl')

        X = x_input.drop(columns=['rla'])

        if model_info['model'] in ['Linear Regression', 'Support Vector Regression', 'Artificial Neural Network']:
            X = scaler_x.transform(X)

        y_pred = model.predict(X)

        y_pred = inv_boxcox(y_pred, fitted_lambda)
        y_pred = y_pred * x_input['rla'].values

        y_pred_ensemble += y_pred

    y_pred_ensemble = y_pred_ensemble / 3
    y_pred_ensemble = y_pred_ensemble[0]
    y_pred_ensemble = round(y_pred_ensemble, 2)

    return y_pred_ensemble



def best_model_predict(x_input):

    model_info = df_top3_models_metrics.iloc[0]

    model = load(f'models/top1_model.pkl')

    X = x_input.drop(columns=['rla'])

    if model_info['model'] in ['Linear Regression', 'Support Vector Regression', 'Artificial Neural Network']:
        X = scaler_x.transform(X)

    y_pred = model.predict(X)

    y_pred = inv_boxcox(y_pred, fitted_lambda)
    y_pred = y_pred * x_input['rla'].values

    y_pred = y_pred[0]

    y_pred = round(y_pred, 2)

    return y_pred



def cask_worth_predict(x_input):

    if models_metrics.iloc[0]['model'] == 'Ensemble Model':
        return ensemble_model_predict(x_input)
    else:
        return best_model_predict(x_input)
import pandas as pd
import streamlit as st

from model_prod import cask_worth_predict

from joblib import load

x_columns_features = load('variables_for_production_model/x_columns_features.pkl')
x_columns_features = [feature.replace('�', '') for feature in x_columns_features]

st.set_page_config(layout='wide')

st.title('Cask Worth')

st.subheader('Whisky Casks Valuation')

st.space('large') 

col_age, col_strength, col_bulk_litres = st.columns(3, vertical_alignment='center')
col_cask_type, col_cask_filling, col_distillery = st.columns(3, vertical_alignment='center')

with col_age:
    age = st.number_input('Age', min_value=0, max_value=100, value=None, step=1)

with col_strength:
    strength = st.number_input('Strength (%)', min_value=40.00, max_value=100.00, value=None, step=0.01)

with col_bulk_litres:
    bulk_litres = st.number_input('Bulk Litres (L)', min_value=0.00, max_value=None, value=None, step=0.01)

with col_distillery:
    distillery = st.text_input('Distillery', value=None)
    distillery = distillery.lower().replace(" ", "_") if (distillery != None) else distillery

with col_cask_type:
    cask_type = st.text_input('Cask Type', value=None)
    cask_type = cask_type.lower().replace(" ", "_") if (cask_type != None) else cask_type

with col_cask_filling:
    cask_filling = st.text_input('Cask Filling', value=None)
    cask_filling = cask_filling.lower().replace(" ", "_") if (cask_filling != None) else cask_filling

st.space('small')

evaluate_button = st.button("Evaluate")

vars_list = [age, strength, bulk_litres, distillery, cask_type, cask_filling]

if (evaluate_button) and all(var is not None for var in vars_list):

    rla = bulk_litres * strength / 100

    dim_distilleries = pd.read_csv('database/dimension/dim_distilleries_info.csv', sep=';')

    distillery_infos = dim_distilleries[dim_distilleries['Distillery'].str.lower().replace(" ", "_") == distillery]
    region = distillery_infos['Region'].values[0] if len(distillery_infos) > 0 else None

    df = pd.DataFrame(columns=x_columns_features)
    df.loc[0, ] = [None] * len(x_columns_features)

    distillery_columns = [f for f in x_columns_features if "distillery" in f]
    df[distillery_columns] = 0

    region_columns = [f for f in x_columns_features if "region" in f]
    df[region_columns] = 0

    cask_type_columns = [f for f in x_columns_features if "cask_type" in f]
    df[cask_type_columns] = 0

    cask_filling_columns = [f for f in x_columns_features if "cask_filling" in f]
    df[cask_filling_columns] = 0

    df['age'] = age
    df['rla'] = rla

    if f'distillery_{distillery}' in df.columns:
        df[f'distillery_{distillery}'] = 1

    if f'region_{region}' in df.columns:
        df[f'region_{region}'] = 1

    if f'cask_type_{cask_type}' in df.columns:
        df[f'cask_type_{cask_type}'] = 1

    if f'cask_filling_{cask_filling}' in df.columns:
        df[f'cask_filling_{cask_filling}'] = 1
    
    casks_value = cask_worth_predict(df)

    st.space('small')

    st.metric('Suggested value', f'£{casks_value:,.2f}')
import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv

from model_prod import cask_worth_predict

load_dotenv()

features = os.getenv("x_columns_all_features").split(",")
features = [feature.replace('�', '') for feature in features]

df_casks = pd.read_csv("database\\casks_database__casks_valuation.csv")
df_casks = df_casks[['auction_date', 'volume_12m', 'volume_6m', 'volume_3m']].drop_duplicates().sort_values('auction_date', ascending=False).head(1)

st.set_page_config(layout='wide')

st.title('Cask Worth')

st.subheader('Whisky Casks Valuation')

st.space('large') 

col_age, col_strength, col_bulk_litres, col_cask_type = st.columns(4, vertical_alignment='center')
col_region, col_country, col_distillery, col_previous_spirit = st.columns(4, vertical_alignment='center')

with col_age:
    age = st.number_input('Age', min_value=0, max_value=100, value=None, step=1)

with col_strength:
    strength = st.number_input('Strength (%)', min_value=40.00, max_value=100.00, value=None, step=0.01)

with col_bulk_litres:
    bulk_litres = st.number_input('Bulk Litres (L)', min_value=0.00, max_value=None, value=None, step=0.01)

with col_distillery:
    distillery = st.text_input('Distillery', value=None)
    distillery = distillery.lower() if (distillery != None) else distillery

with col_region:
    region = st.text_input('Region', value=None)
    region = region.lower() if (region != None) else region

with col_country:
    country = st.text_input('Country', value=None)
    country = country.lower() if (country != None) else country

with col_cask_type:
    cask_type = st.text_input('Cask Type', value=None)
    cask_type = cask_type.lower() if (cask_type != None) else cask_type

with col_previous_spirit:
    previous_spirit = st.text_input('Previous Spirit', value=None)
    previous_spirit = previous_spirit.lower() if (previous_spirit != None) else previous_spirit

st.space('small')

evaluate_button = st.button("Evaluate")

vars_list = [age, strength, bulk_litres, distillery, region, country, cask_type, previous_spirit]

if (evaluate_button) and all(var is not None for var in vars_list):

    rla = bulk_litres * strength / 100
    bottles_at_cask_strength = bulk_litres / 0.7

    df = pd.DataFrame(columns=features)
    df.loc[0, ] = [None] * len(features)

    distillery_columns = [f for f in features if "distillery" in f]
    df[distillery_columns] = 0

    region_columns = [f for f in features if "region" in f]
    df[region_columns] = 0

    country_columns = [f for f in features if "country" in f]
    df[country_columns] = 0

    cask_type_columns = [f for f in features if "cask_type" in f]
    df[cask_type_columns] = 0

    previous_spirit_columns = [f for f in features if "previous_spirit" in f]
    df[previous_spirit_columns] = 0

    df['age'] = age
    df['strength'] = strength
    df['bulk_litres'] = bulk_litres
    df['rla'] = rla
    df['bottles_at_cask_strength'] = bottles_at_cask_strength

    df['volume_12m'] = df_casks['volume_12m'].values[0]
    df['volume_6m'] = df_casks['volume_6m'].values[0]
    df['volume_3m'] = df_casks['volume_3m'].values[0]

    if f'distillery_{distillery}' in df.columns:
        df[f'distillery_{distillery}'] = 1
    if f'region_{region}' in df.columns:
        df[f'region_{region}'] = 1
    if f'country_{country}' in df.columns:
        df[f'country_{country}'] = 1
    if f'cask_type_{cask_type}' in df.columns:
        df[f'cask_type_{cask_type}'] = 1
    if f'previous_spirit_{previous_spirit}' in df.columns:
        df[f'previous_spirit_{previous_spirit}'] = 1
    
    casks_value = cask_worth_predict(df)

    st.space('small')

    st.metric('Suggested value', f'£{casks_value:,.2f}')
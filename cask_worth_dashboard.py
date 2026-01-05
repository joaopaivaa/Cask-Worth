import streamlit as st

st.set_page_config(layout='wide')

st.title('Cask Worth - Whisky Casks Valuation')

st.space('large')

col_age, col_strength, col_bulk_litres = st.columns(3, vertical_alignment='center')
col_region, col_country, col_distillery = st.columns(3, vertical_alignment='center')

with col_age:
    age = st.number_input('Age', min_value=0, max_value=100, value=None, step=1)

with col_strength:
    strength = st.number_input('Strength (%)', min_value=40.00, max_value=100.00, value=None, step=0.01)

with col_bulk_litres:
    bulk_litres = st.number_input('Bulk Litres (L)', min_value=0.00, max_value=None, value=None, step=0.01)

with col_distillery:
    distillery = st.text_input('Distillery', value=None)
    distillery = distillery.str.lower()

with col_region:
    region = st.text_input('Region', value=None)
    region = region.str.lower()

with col_country:
    country = st.text_input('Country', value=None)
    country = country.str.lower()











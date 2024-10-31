import streamlit as st
import polars as pl
import pandas as pd

import requests
import json
import logging
import os
import time

from datetime import datetime
from page_renders.landing import render_main_page
from page_renders.visuals import render_dashboards
from page_renders.utils import add_prediction_column

st.set_page_config(
    page_title="m6a prediction dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    # page_icon="./assets/image.png"
)

PAGES = {
    "Home": render_main_page,
    "Visuals": render_dashboards,
}

st.sidebar.title("Navigation")
selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))

def load_data():
    folder_path = "data/parquet_files"
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]
    
    data_frames = [pl.read_parquet(file) for file in files]
    combined_df = pl.concat(data_frames)
    return combined_df


if 'data' not in st.session_state or st.sidebar.button("Reload Data"):
    with st.spinner("Loading data..."):
        st.session_state['data'] = add_prediction_column(load_data())
        st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    success = st.success("Data loaded successfully.")
    time.sleep(2)
    success.empty()


st.sidebar.markdown(f"**Last Update:** {st.session_state.get('last_update', 'Not loaded')}")

page = PAGES[selection]
page()
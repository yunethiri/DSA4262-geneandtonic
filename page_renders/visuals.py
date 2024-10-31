import streamlit as st
import requests
import json
import logging

from page_renders.utils import analyze_numerical_mutations

def render_dashboards():
    # Configure logging
    # logging.basicConfig(level=logging.DEBUG, filename='chatbot.log', filemode='w', format='%(asctime)s - %(message)s')

    st.title("m6a Predictions Dashboard")

    df = st.session_state['data']

    st.markdown('## Numerical Analysis')
    analyze_numerical_mutations(df)

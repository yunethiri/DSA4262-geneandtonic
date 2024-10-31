import streamlit as st
import polars as pl

def get_unique_cell_lines(df):
    """Get unique cell lines from the DataFrame."""
    unique_cell_lines = df.select('cell_line').unique()
    return unique_cell_lines.to_pandas() 

def get_run_counts(df):
    """Count 1s and 0s for each run and calculate the proportion of 1s."""
    run_counts = df.group_by(['run', 'prediction']).agg(pl.count().alias("count"))
    
    run_counts = run_counts.pivot(
        values="count",
        index="run",
        columns="prediction"
    ).fill_null(0)
    
    run_counts = run_counts.rename({"run": "Cell Line Run", "0": "0 (No Mutation)", "1": "1 (Mutation)"})
    
    run_counts = run_counts.with_columns(
        ((run_counts['1 (Mutation)'] / (run_counts['0 (No Mutation)'] + run_counts['1 (Mutation)'])) * 100)
        .round(2)
        .alias("Proportion of Mutations (%)")
    )
    
    return run_counts.to_pandas()  

def render_main_page():
    st.title("m6a Predictions Dashboard")
    st.write("Main page for analysis, displayed here are the Cell Lines and a brief overview of the predicitons on each Run across the Cell Lines")


    df = st.session_state['data']
    
    c1, c2 = st.columns((0.3, 0.7))
    
    c1.subheader("Unique Cell Lines")
    unique_cell_lines = get_unique_cell_lines(df)
    c1.write(unique_cell_lines)
    
    run_counts = get_run_counts(df)
    
    c2.subheader("Number of Mutations for Each Unique Run")
    c2.write(run_counts)

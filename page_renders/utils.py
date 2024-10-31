import streamlit as st
import polars as pl

def add_prediction_column(df):
    """Add a prediction column based on the score threshold of 0.5."""
    return df.with_columns((df['score'] > 0.5).cast(pl.Int8).alias('prediction'))

import plotly.graph_objects as go

def analyze_numerical_mutations(df):
    """Analyze and plot trends in mutations and non-mutations with user-selected columns for analysis and grouping."""
    
    c1,c2,c3,c4 = st.columns((0.25,0.2,0.2,0.35))

    # Filter to only numerical columns for x-axis selection
    numerical_columns = [col for col in df.columns if df.schema[col] in [pl.Float64, pl.Int64]]
    x_axis_column = c1.selectbox("Select numerical feature to compare by", numerical_columns)
    group_by_column = c2.selectbox("Group by: (Run/ Cell Line)", ["run", "cell_line"])

    df = df.with_columns((pl.col("cell_line") + "_" + pl.col("run")).alias("combined_run"))
    if group_by_column == "run":
        group_by_column = "combined_run"  # Update group_by_column to the new combined column

    # Filter by which cell lines/ runs to include
    unique_runs = df.select("combined_run").unique().to_series().to_list() if group_by_column == "combined_run" else df.select("cell_line").unique().to_series().to_list()
    selected_runs = st.multiselect("Select Runs or Cell Lines to Include", unique_runs, default=unique_runs)
    df = df.filter(pl.col(group_by_column).is_in(selected_runs))
    
    # Purely for handling latency on graph generation, no binning displays counts for all value-types
    # Check the number of unique values in x-axis column
    unique_values_count = df.select(x_axis_column).n_unique()
    column_min = df.select(x_axis_column).min()[0, 0]
    column_max = df.select(x_axis_column).max()[0, 0]
    unique_values_threshold = 5000  
    
    # Toggle binning, with an automatic enable if unique values exceed threshold
    enable_binning = c3.checkbox(f"Enable binning for {group_by_column}", value=unique_values_count > unique_values_threshold)

    if enable_binning:
        # Dynamic binning based on x-values
        max_bin = max(1.0, (column_max - column_min) / 10)  # Set max_bin based on data range
        step_bin = max(0.01, max_bin / 100)  # Dynamic step size based on max_bin
        default_bin = max(1.0, max_bin / 5)

        # Slider for adjusting
        bin_size = c4.slider(
            f"Select bin size for {group_by_column}",
            min_value=0.001,
            max_value=max_bin,
            value=default_bin,
            step=step_bin
        )
        # Bin x-axis values, supporting decimal bin sizes
        df = df.with_columns(((pl.col(x_axis_column) / bin_size).floor() * bin_size).alias(f"{x_axis_column}_binned"))
        x_axis_column_binned = f"{x_axis_column}_binned"
    else:
        x_axis_column_binned = x_axis_column  # Use original column if binning is disabled
    
    # Group by `x_axis_column_binned`, `group_by_column`, and `prediction`, then aggregate counts
    mutation_counts = (
        df.group_by([x_axis_column_binned, group_by_column, "prediction"])
        .agg(pl.count().alias("count"))
        .sort([x_axis_column_binned, group_by_column])
    )
    
    # Generation of visuals
    mutation_counts = mutation_counts.to_pandas()
    fig = go.Figure()

    for group in mutation_counts[group_by_column].unique():
        group_data = mutation_counts[mutation_counts[group_by_column] == group]
        
        mutation_data = group_data[group_data["prediction"] == 1]
        non_mutation_data = group_data[group_data["prediction"] == 0]
        
        fig.add_trace(
            go.Scatter(
                x=mutation_data[x_axis_column_binned],
                y=mutation_data["count"],
                mode="lines+markers",
                name=f"{group} - Mutation",
                line=dict(width=2, color="firebrick"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=non_mutation_data[x_axis_column_binned],
                y=non_mutation_data["count"],
                mode="lines+markers",
                name=f"{group} - No Mutation",
                line=dict(width=2, dash="dash", color="royalblue"),
            )
        )
    
    fig.update_layout(
        title=f"Mutation and Non-mutation Trends across {x_axis_column.capitalize()} by {group_by_column.capitalize()}",
        xaxis_title=x_axis_column.capitalize(),
        yaxis_title="Count",
        legend_title=group_by_column.capitalize(),
        template="plotly_dark",
    )
    
    st.plotly_chart(fig, use_container_width=True)

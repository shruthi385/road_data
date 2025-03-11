import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import random
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
import base64
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Frame, KeepInFrame

# Set random seed for reproducibility
random_seed = 42

# Set Streamlit page configuration
st.set_page_config(
    page_title="Project Cost Overrun Prediction Dashboard",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling and FontAwesome icons
st.markdown(
    """
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

    .main {
        background-color: #f0f2f6;
    }

    .icon {
        font-size: 20px;
        margin-right: 10px;
    }

    /* Styled metric boxes */
    .stMetric {
        background-color: blue;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
        color: white;
    }

    .stMetric:nth-child(2n) {
        background-color: #f1f8e9;
        color: black;
    }

    .stMetric:nth-child(3n) {
        background-color: #ffecb3;
        color: black;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Welcome Section
st.markdown('<h1><i class="fa-solid fa-road"></i> Project Cost Overrun Prediction Dashboard</h1>', unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Progress Bar for File Upload
    with st.spinner("Uploading and processing your file..."):
        time.sleep(2)  # Simulate processing time
    st.success("File uploaded and processed successfully!")

    # Load the dataset
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            df = pd.read_excel(uploaded_file)
    except UnicodeDecodeError:
        st.error("There was an error reading the file. Please make sure the file encoding is correct.")
        st.stop()

    # Data Preview
    st.write("### Data Preview")
    st.write("This section shows the first few rows of the uploaded dataset. It provides a quick overview of the data structure and content.")

    # Toggle for Raw Data Display
    if st.checkbox("Show Raw Data"):
        st.write("### Raw Data")
        st.write(df)

    # Display the data in a styled table
    st.dataframe(df.head().style
                .set_properties(**{'background-color': '#f0f2f6', 'color': 'black', 'border': '1px solid black'})
                .set_table_styles([{'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white')]}]))

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Feature Engineering
    df['Letter Of Award'] = pd.to_datetime(df['Letter Of Award'], errors='coerce')
    df['Construction Commencement'] = pd.to_datetime(df['Construction Commencement'], errors='coerce')
    df['Scheduled Project Completion'] = pd.to_datetime(df['Scheduled Project Completion'], errors='coerce')
    df['Actual Completion'] = pd.to_datetime(df['Actual Completion'], errors='coerce')

    df['Construction Period (Months)'] = ((df['Construction Commencement'] - df['Letter Of Award']).dt.days / 30).astype(int)
    df['Time Taken for Completion'] = (df['Actual Completion'] - df['Construction Commencement']).dt.days
    df['Scheduled Time for Completion'] = (df['Scheduled Project Completion'] - df['Construction Commencement']).dt.days
    df['Delay Period (Days)'] = df['Time Taken for Completion'] - df['Scheduled Time for Completion']

    df['Delay Period (Months)'] = df['Delay Period (Days)'] / 30

    # Data inspection
    st.write("### Data Inspection")
    st.write("This section provides a statistical summary of key numerical columns in the dataset. It helps you understand the distribution and range of the data.")
    st.write(df[['Construction Period (Months)', 'Stretch (Kms)', 'Lane', 'Project Cost (Rs.Cr)', 'Revised Project Cost (Rs.Cr)', 'Incremental Cost (Rs.Cr)', 'Delay Period (Months)']].describe())

    # Count metrics
    total_projects = df.shape[0]
    unique_states = df['State'].nunique() if 'State' in df.columns else 0

    # Dynamically count project statuses
    project_status_counts = df['Project Status'].value_counts().reset_index()
    project_status_counts.columns = ['Project Status', 'Count']

    # Display count boxes
    st.write("### Metrics Overview")
    st.write("This section provides key metrics about the dataset, such as the total number of projects and unique states.")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Projects", value=total_projects)
    with col2:
        st.metric(label="Unique States", value=unique_states)

    # Display dynamic project status counts
    st.write("### Project Status Counts")
    st.write("This bar chart shows the distribution of projects by their status. It helps you understand how many projects are completed, delayed, or in progress.")
    fig_status = px.bar(
        project_status_counts,
        x='Project Status',
        y='Count',
        color='Project Status',
        title="Project Status Distribution",
        labels={'Count': 'Number of Projects', 'Project Status': 'Status'}
    )
    fig_status.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis_title="Project Status",
        yaxis_title="Number of Projects"
    )
    st.plotly_chart(fig_status)

    # Sidebar filters
    st.sidebar.header("🔍 Filter Options")
    st.sidebar.write("Use the filters below to narrow down the dataset based on specific criteria.")

    # Project Commencement Year Filter
    all_years = df['Construction Commencement'].dt.year.unique().tolist()
    year_filter = st.sidebar.multiselect(
        "Project Commencement Year", 
        options=all_years, 
        default=all_years
    )
    if len(year_filter) == len(all_years):
        year_filter_label = "All Years Selected"
    else:
        year_filter_label = ", ".join(str(year) for year in year_filter)
    st.sidebar.write(f"Selected Years: {year_filter_label}")

    # Lane Configuration Filter
    all_lanes = df['Lane'].unique().tolist()
    lane_filter = st.sidebar.multiselect(
        "Lane Configuration", 
        options=all_lanes, 
        default=all_lanes
    )
    if len(lane_filter) == len(all_lanes):
        lane_filter_label = "All Lanes Selected"
    else:
        lane_filter_label = ", ".join(str(lane) for lane in lane_filter)
    st.sidebar.write(f"Selected Lanes: {lane_filter_label}")

    # Project Mode Filter
    all_modes = df['Mode'].unique().tolist()
    mode_filter = st.sidebar.multiselect(
        "Project Mode", 
        options=all_modes, 
        default=all_modes
    )
    if len(mode_filter) == len(all_modes):
        mode_filter_label = "All Modes Selected"
    else:
        mode_filter_label = ", ".join(mode_filter)
    st.sidebar.write(f"Selected Modes: {mode_filter_label}")

    # State Filter
    all_states = df['State'].unique().tolist()
    state_filter = st.sidebar.multiselect(
        "State", 
        options=all_states, 
        default=all_states
    )
    if len(state_filter) == len(all_states):
        state_filter_label = "All States Selected"
    else:
        state_filter_label = ", ".join(state_filter)
    st.sidebar.write(f"Selected States: {state_filter_label}")

    # Construction Duration Filter
    duration_filter = st.sidebar.slider(
        "Construction Duration (Months)", 
        min_value=int(df['Construction Period (Months)'].min()), 
        max_value=int(df['Construction Period (Months)'].max()), 
        value=(int(df['Construction Period (Months)'].min()), int(df['Construction Period (Months)'].max()))
    )

    # Concession Period Filter
    concession_filter = st.sidebar.slider(
        "Concession Period (Years)", 
        min_value=int(df['Concession Period (Years)'].min()), 
        max_value=int(df['Concession Period (Years)'].max()), 
        value=(int(df['Concession Period (Years)'].min()), int(df['Concession Period (Years)'].max()))
    )

    # Stretch Distance Filter
    distance_filter = st.sidebar.slider(
        "Stretch Distance (Kms)", 
        min_value=float(df['Stretch (Kms)'].min()), 
        max_value=float(df['Stretch (Kms)'].max()), 
        value=(float(df['Stretch (Kms)'].min()), float(df['Stretch (Kms)'].max()))
    )

    # Applying filters
    filtered_df = df[
        (df['Construction Commencement'].dt.year.isin(year_filter)) &
        (df['Lane'].isin(lane_filter)) &
        (df['Mode'].isin(mode_filter)) &
        (df['State'].isin(state_filter)) &
        (df['Construction Period (Months)'] >= duration_filter[0]) & 
        (df['Construction Period (Months)'] <= duration_filter[1]) &
        (df['Concession Period (Years)'] >= concession_filter[0]) & 
        (df['Concession Period (Years)'] <= concession_filter[1]) &
        (df['Stretch (Kms)'] >= distance_filter[0]) & 
        (df['Stretch (Kms)'] <= distance_filter[1])
    ]

    st.write("### Filtered Data Preview")
    st.write("This section shows the first few rows of the dataset after applying the selected filters. It helps you verify that the filters are working as expected.")

    # Download Button for Filtered Data
    if st.button("Download Filtered Data as CSV"):
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )

    # Display the filtered data in a styled table
    st.dataframe(filtered_df.head().style
                .set_properties(**{'background-color': '#f0f2f6', 'color': 'black', 'border': '1px solid black'})
                .set_table_styles([{'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white')]}]))

    # Select relevant features for cost overrun prediction
    cost_overrun_features = ['Project Cost (Rs.Cr)', 'Incremental Cost (Rs.Cr)', 'Construction Period (Months)', 'Delay Period (Months)']
    cost_overrun_target = 'Revised Project Cost (Rs.Cr)'

    X_cost_overrun = filtered_df[cost_overrun_features]
    y_cost_overrun = filtered_df[cost_overrun_target]

    # Train-Test Split for cost overrun prediction
    X_train_cost, X_test_cost, y_train_cost, y_test_cost = train_test_split(X_cost_overrun, y_cost_overrun, test_size=0.2, random_state=random_seed)

    # Regression Model for cost overrun prediction
    cost_model = RandomForestRegressor(random_state=random_seed)
    cost_model.fit(X_train_cost, y_train_cost)

    # Predictions for cost overrun
    y_pred_cost = cost_model.predict(X_test_cost)

    # Evaluation for cost overrun prediction
    st.write("### Model Evaluation for Cost Overrun Prediction")
    st.write("This section evaluates the performance of the prediction model. The **Mean Squared Error (MSE)** measures the average squared difference between the actual and predicted revised project costs. A lower MSE indicates better model performance.")

    # Calculate MSE for cost overrun prediction
    mse_cost = mean_squared_error(y_test_cost, y_pred_cost)

    # Display MSE in a metric card
    st.metric(label="Mean Squared Error (MSE) for Cost Overrun Prediction", value=f"{mse_cost:,.2f}")

    # Input form for cost overrun prediction
    st.write("### Predict Project Cost Overrun")
    st.write("Enter the project details below to predict the revised project cost.")

    # Input fields for cost overrun prediction
    project_cost = st.number_input("Project Cost (Rs.Cr)", min_value=0.0, value=100.0, help="Enter the initial project cost in Rs.Cr.")
    incremental_cost = st.number_input("Incremental Cost (Rs.Cr)", min_value=0.0, value=10.0, help="Enter the incremental cost in Rs.Cr.")
    construction_period = st.number_input("Construction Period (Months)", min_value=0, value=24, help="Enter the planned construction period in months.")
    delay_period = st.number_input("Delay Period (Months)", min_value=0.0, value=6.0, help="Enter the delay period in months.")

    if st.button("Predict Cost Overrun"):
        # Make prediction for cost overrun
        prediction_cost = cost_model.predict([[project_cost, incremental_cost, construction_period, delay_period]])[0]

        # Calculate cost overrun
        cost_overrun = prediction_cost - project_cost

        # Display prediction result for cost overrun
        if cost_overrun > 0:
            st.warning(f"The project is predicted to exceed the initial budget by **{cost_overrun:,.2f} Rs.Cr**.")
            st.write(f"The revised project cost is predicted to be **{prediction_cost:,.2f} Rs.Cr**.")
        else:
            st.success(f"The project is predicted to be within the initial budget.")
            st.write(f"The revised project cost is predicted to be **{prediction_cost:,.2f} Rs.Cr**.")

        # Set a session state variable to indicate that prediction has been made
        st.session_state['prediction_cost_made'] = True
        st.session_state['prediction_cost'] = prediction_cost

        # Visualization for cost overrun prediction
        st.write("### Visualization of Predictions vs Actual for Cost Overrun")

        # Scatter plot of actual vs predicted revised project costs
        st.write("**Actual vs Predicted Revised Project Cost**")
        st.write("Each point represents a project. The x-axis shows the actual revised project cost, and the y-axis shows the predicted revised project cost. Points close to the diagonal line indicate accurate predictions.")

        # Create a dataframe for the scatter plot
        scatter_df_cost = pd.DataFrame({
            'Actual Revised Project Cost (Rs.Cr)': y_test_cost,
            'Predicted Revised Project Cost (Rs.Cr)': y_pred_cost
        })

        # Plot the scatter plot with a diagonal line
        fig_scatter_cost = px.scatter(
            scatter_df_cost, 
            x='Actual Revised Project Cost (Rs.Cr)', 
            y='Predicted Revised Project Cost (Rs.Cr)', 
            title="Actual vs Predicted Revised Project Cost (Rs.Cr)",
            labels={"Actual Revised Project Cost (Rs.Cr)": "Actual Revised Project Cost (Rs.Cr)", "Predicted Revised Project Cost (Rs.Cr)": "Predicted Revised Project Cost (Rs.Cr)"}
        )
        fig_scatter_cost.add_shape(
            type="line",
            x0=scatter_df_cost['Actual Revised Project Cost (Rs.Cr)'].min(),
            y0=scatter_df_cost['Actual Revised Project Cost (Rs.Cr)'].min(),
            x1=scatter_df_cost['Actual Revised Project Cost (Rs.Cr)'].max(),
            y1=scatter_df_cost['Actual Revised Project Cost (Rs.Cr)'].max(),
            line=dict(color="red", width=2, dash="dash")
        )
        fig_scatter_cost.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            xaxis_title="Actual Revised Project Cost (Rs.Cr)",
            yaxis_title="Predicted Revised Project Cost (Rs.Cr)",
            hovermode="closest",
            showlegend=False
        )
        st.plotly_chart(fig_scatter_cost)

        # Line chart of actual vs predicted revised project costs over time
        st.write("**Actual vs Predicted Revised Project Cost Over Time**")
        st.write("This line chart compares the actual and predicted revised project costs over time. The blue line represents actual revised project costs, and the red line represents predicted revised project costs.")

        # Create a dataframe for the line chart
        comparison_df_cost = pd.DataFrame({
            'Year': df.loc[X_test_cost.index, 'Construction Commencement'].dt.year,  # Use actual years from the dataset
            'Actual Revised Project Cost (Rs.Cr)': y_test_cost,
            'Predicted Revised Project Cost (Rs.Cr)': y_pred_cost
        })

        # Group by year to smooth the lines
        comparison_df_cost = comparison_df_cost.groupby('Year').mean().reset_index()

        # Plot the line chart
        fig_line_cost = px.line(
            comparison_df_cost, 
            x='Year', 
            y=['Actual Revised Project Cost (Rs.Cr)', 'Predicted Revised Project Cost (Rs.Cr)'], 
            title="Actual vs Predicted Revised Project Cost Over Time (Rs.Cr)",
            labels={"value": "Revised Project Cost (Rs.Cr)", "variable": "Type"}
        )
        fig_line_cost.update_traces(line=dict(width=3))
        fig_line_cost.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            xaxis_title="Year",
            yaxis_title="Revised Project Cost (Rs.Cr)",
            legend_title="Type",
            hovermode="x unified"        )
        st.plotly_chart(fig_line_cost)

        # Pie chart of state distribution
        st.write("**Project State Distribution**")
        st.write("This pie chart shows the distribution of projects by state. It helps to understand the proportion of different states in the dataset.")
        fig_pie = px.pie(filtered_df, names="State", title="Project State Distribution")
        fig_pie.update_traces(textinfo='percent+label')
        fig_pie.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )
        st.plotly_chart(fig_pie)

        # Histogram of project costs
        st.write("**Distribution of Initial Project Costs**")
        st.write("This histogram shows the distribution of initial project costs across all projects.")
        fig_hist_project_cost = px.histogram(filtered_df, x="Project Cost (Rs.Cr)", nbins=50, title="Distribution of Initial Project Costs (Rs.Cr)")
        fig_hist_project_cost.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            xaxis_title="Project Cost (Rs.Cr)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_hist_project_cost)

        # Histogram of revised project costs
        st.write("**Distribution of Revised Project Costs**")
        st.write("This histogram shows the distribution of revised project costs across all projects.")
        fig_hist_revised_cost = px.histogram(filtered_df, x="Revised Project Cost (Rs.Cr)", nbins=50, title="Distribution of Revised Project Costs (Rs.Cr)")
        fig_hist_revised_cost.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            xaxis_title="Revised Project Cost (Rs.Cr)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_hist_revised_cost)

        # Bar chart of project costs by mode
        st.write("**Project Costs by Mode**")
        st.write("This bar chart shows the average initial project costs by project mode.")
        avg_cost_by_mode = filtered_df.groupby('Mode')['Project Cost (Rs.Cr)'].mean().reset_index()
        fig_bar_cost_mode = px.bar(
            avg_cost_by_mode,
            x='Mode',
            y='Project Cost (Rs.Cr)',
            title='Average Initial Project Cost by Mode',
            labels={'Project Cost (Rs.Cr)': 'Average Project Cost (Rs.Cr)', 'Mode': 'Project Mode'}
        )
        fig_bar_cost_mode.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            xaxis_title="Project Mode",
            yaxis_title="Average Project Cost (Rs.Cr)"
        )
        st.plotly_chart(fig_bar_cost_mode)

        # Bar chart of revised project costs by mode
        st.write("**Revised Project Costs by Mode**")
        st.write("This bar chart shows the average revised project costs by project mode.")
        avg_revised_cost_by_mode = filtered_df.groupby('Mode')['Revised Project Cost (Rs.Cr)'].mean().reset_index()
        fig_bar_revised_cost_mode = px.bar(
            avg_revised_cost_by_mode,
            x='Mode',
            y='Revised Project Cost (Rs.Cr)',
            title='Average Revised Project Cost by Mode',
            labels={'Revised Project Cost (Rs.Cr)': 'Average Revised Project Cost (Rs.Cr)', 'Mode': 'Project Mode'}
        )
        fig_bar_revised_cost_mode.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            xaxis_title="Project Mode",
            yaxis_title="Average Revised Project Cost (Rs.Cr)"
        )
        st.plotly_chart(fig_bar_revised_cost_mode)

        # Area chart of project costs over time
        st.write("**Project Costs Over Time**")
        st.write("This area chart shows the trend of initial project costs over time.")
        df['Year'] = df['Construction Commencement'].dt.year
        cost_over_time = df.groupby('Year')['Project Cost (Rs.Cr)'].mean().reset_index()
        fig_area_cost_time = px.area(
            cost_over_time,
            x='Year',
            y='Project Cost (Rs.Cr)',
            title='Average Initial Project Cost Over Time',
            labels={'Project Cost (Rs.Cr)': 'Average Project Cost (Rs.Cr)', 'Year': 'Year'}
        )
        fig_area_cost_time.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            xaxis_title="Year",
            yaxis_title="Average Project Cost (Rs.Cr)"
        )
        st.plotly_chart(fig_area_cost_time)

        # Area chart of revised project costs over time
        st.write("**Revised Project Costs Over Time**")
        st.write("This area chart shows the trend of revised project costs over time.")
        revised_cost_over_time = df.groupby('Year')['Revised Project Cost (Rs.Cr)'].mean().reset_index()
        fig_area_revised_cost_time = px.area(
            revised_cost_over_time,
            x='Year',
            y='Revised Project Cost (Rs.Cr)',
            title='Average Revised Project Cost Over Time',
            labels={'Revised Project Cost (Rs.Cr)': 'Average Revised Project Cost (Rs.Cr)', 'Year': 'Year'}
        )
        fig_area_revised_cost_time.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            xaxis_title="Year",
            yaxis_title="Average Revised Project Cost (Rs.Cr)"
        )
        st.plotly_chart(fig_area_revised_cost_time)

        # Heatmap of correlation between features
        st.write("**Correlation Between Features**")
        st.write("This heatmap shows the correlation between different numerical features in the dataset.")
        correlation_matrix = filtered_df[['Project Cost (Rs.Cr)', 'Revised Project Cost (Rs.Cr)', 'Incremental Cost (Rs.Cr)', 'Construction Period (Months)', 'Delay Period (Months)']].corr()
        fig_heatmap = px.imshow(
            correlation_matrix,
            title='Correlation Between Features',
            labels={'color': 'Correlation'},
            x=['Project Cost (Rs.Cr)', 'Revised Project Cost (Rs.Cr)', 'Incremental Cost (Rs.Cr)', 'Construction Period (Months)', 'Delay Period (Months)'],
            y=['Project Cost (Rs.Cr)', 'Revised Project Cost (Rs.Cr)', 'Incremental Cost (Rs.Cr)', 'Construction Period (Months)', 'Delay Period (Months)']
        )
        fig_heatmap.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )
        st.plotly_chart(fig_heatmap)
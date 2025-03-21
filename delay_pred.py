import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
from prophet import Prophet
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
    page_title="Project Delay Prediction Dashboard",
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
st.markdown('<h1><i class="fa-solid fa-road"></i> Project Delay Prediction Dashboard</h1>', unsafe_allow_html=True)

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

    # Convert delay periods to months and years for better understanding
    df['Delay Period (Months)'] = df['Delay Period (Days)'] / 30
    df['Delay Period (Years)'] = df['Delay Period (Days)'] / 365

    df = df.dropna(subset=['Delay Period (Days)'])

    # Data inspection
    st.write("### Data Inspection")
    st.write("This section provides a statistical summary of key numerical columns in the dataset. It helps you understand the distribution and range of the data.")
    st.write(df[['Construction Period (Months)', 'Stretch (Kms)', 'Lane', 'Delay Period (Months)', 'Delay Period (Years)']].describe())

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

    # Select relevant features
    features = ['Construction Period (Months)', 'Stretch (Kms)', 'Lane']
    target = 'Delay Period (Days)'

    # Convert categorical features to numerical
    filtered_df['Lane'] = filtered_df['Lane'].astype('category').cat.codes

    X = filtered_df[features]
    y = filtered_df[target]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Regression Model
    model = RandomForestRegressor(random_state=random_seed)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    st.write("### Model Evaluation")
    st.write("This section evaluates the performance of the prediction model. The **Mean Squared Error (MSE)** measures the average squared difference between the actual and predicted delay periods. A lower MSE indicates better model performance.")

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)

    # Display MSE in a metric card
    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:,.2f}")

    # Input form for prediction
    st.write("### Predict Project Delay")
    st.write("Enter the project details below to predict the delay.")

    # Input fields
    construction_period = st.number_input("Construction Period (Months)", min_value=0, value=24, help="Enter the planned construction period in months.")
    stretch_km = st.number_input("Stretch Distance (Kms)", min_value=0.0, value=45.88, help="Enter the length of the road stretch in kilometers.")
    lane = st.number_input("Lane Configuration", min_value=1, value=2, help="Enter the number of lanes (e.g., 2 for 2-lane road).")

    if st.button("Predict Delay"):
        # Make prediction
        prediction = model.predict([[construction_period, stretch_km, lane]])[0]

        # Convert days to months and years for better understanding
        prediction_months = prediction / 30
        prediction_years = prediction / 365

        # Display prediction result
        if prediction < 0:
            st.success(f"The project is predicted to be completed **{abs(prediction):,.0f} days** earlier than scheduled.")
            st.write(f"This is approximately **{abs(prediction_months):,.1f} months** or **{abs(prediction_years):,.1f} years** earlier.")
        else:
            st.warning(f"The project is predicted to be delayed by **{prediction:,.0f} days**.")
            st.write(f"This is approximately **{prediction_months:,.1f} months** or **{prediction_years:,.1f} years**.")

        # Calculate the expected completion date
        loa_date = pd.to_datetime('today')  # Assuming today's date as LOA date for prediction
        expected_completion_date = loa_date + pd.DateOffset(months=construction_period) + pd.DateOffset(days=prediction)
        st.write(f"**Expected Completion Date:** {expected_completion_date.date()}")

        # Set a session state variable to indicate that prediction has been made
        st.session_state['prediction_made'] = True
        st.session_state['prediction'] = prediction

    # Visualization
    st.write("### Visualization of Predictions vs Actual")

    # Delay Trend Over Time
    df['Year'] = df['Construction Commencement'].dt.year
    delay_trend = df.groupby('Year')['Delay Period (Months)'].mean().reset_index()

    st.write("### Average Delay Trend Over Time")
    st.write("This line chart shows the average delay trend over time. It helps you understand how delays have changed over the years.")
    fig_trend = px.line(delay_trend, x='Year', y='Delay Period (Months)', title="Average Delay Trend Over Time (in Months)")
    fig_trend.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis_title="Year",
        yaxis_title="Average Delay (Months)"
    )
    st.plotly_chart(fig_trend)

    # Scatter plot of actual vs predicted delay periods
    st.write("**Actual vs Predicted Delay Period**")
    st.write("Each point represents a project. The x-axis shows the actual delay period, and the y-axis shows the predicted delay period. Points close to the diagonal line indicate accurate predictions.")

    # Create a dataframe for the scatter plot
    scatter_df = pd.DataFrame({
        'Actual Delay (Months)': y_test / 30,
        'Predicted Delay (Months)': y_pred / 30
    })

    # Plot the scatter plot with a diagonal line
    fig_scatter = px.scatter(
        scatter_df, 
        x='Actual Delay (Months)', 
        y='Predicted Delay (Months)', 
        title="Actual vs Predicted Delay Period (in Months)",
        labels={"Actual Delay (Months)": "Actual Delay (Months)", "Predicted Delay (Months)": "Predicted Delay (Months)"}
    )
    fig_scatter.add_shape(
        type="line",
        x0=scatter_df['Actual Delay (Months)'].min(),
        y0=scatter_df['Actual Delay (Months)'].min(),
        x1=scatter_df['Actual Delay (Months)'].max(),
        y1=scatter_df['Actual Delay (Months)'].max(),
        line=dict(color="red", width=2, dash="dash")
    )
    fig_scatter.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis_title="Actual Delay (Months)",
        yaxis_title="Predicted Delay (Months)",
        hovermode="closest",
        showlegend=False
    )
    st.plotly_chart(fig_scatter)

    # Line chart of actual vs predicted delay periods over time
    st.write("**Actual vs Predicted Delay Period Over Time**")
    st.write("This line chart compares the actual and predicted delay periods over time. The blue line represents actual delays, and the red line represents predicted delays.")

    # Create a dataframe for the line chart
    comparison_df = pd.DataFrame({
        'Year': df.loc[X_test.index, 'Construction Commencement'].dt.year,  # Use actual years from the dataset
        'Actual Delay (Months)': y_test / 30,
        'Predicted Delay (Months)': y_pred / 30
    })

    # Group by year to smooth the lines
    comparison_df = comparison_df.groupby('Year').mean().reset_index()

    # Plot the line chart
    fig_line = px.line(
        comparison_df, 
        x='Year', 
        y=['Actual Delay (Months)', 'Predicted Delay (Months)'], 
        title="Actual vs Predicted Delay Period Over Time (in Months)",
        labels={"value": "Delay Period (Months)", "variable": "Type"}
    )
    fig_line.update_traces(line=dict(width=3))
    fig_line.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis_title="Year",
        yaxis_title="Delay Period (Months)",
        legend_title="Type",
        hovermode="x unified"
    )
    st.plotly_chart(fig_line)

    # Histogram of delay periods
    st.write("**Distribution of Delay Periods**")
    st.write("This histogram shows how often different delay periods occur across all projects.")
    fig_hist = px.histogram(filtered_df, x="Delay Period (Months)", nbins=50, title="Distribution of Delay Periods (in Months)")
    fig_hist.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis_title="Delay Period (Months)",
        yaxis_title="Frequency",
        xaxis=dict(
            tickmode='array',
            tickvals=[-100, -50, 0, 50, 100, 150, 200],
            ticktext=['-100', '-50', '0', '50', '100', '150', '200']
        )
    )
    st.plotly_chart(fig_hist)

    # Scatter plot of stretch km vs delay period
    st.write("**Stretch Distance vs Delay Period**")
    st.write("This scatter plot shows the relationship between the length of the project stretch and the delay period. Different colors represent different lane configurations.")
    fig_scatter = px.scatter(
        filtered_df, 
        x="Stretch (Kms)", 
        y="Delay Period (Months)", 
        color="Lane",
        title="Stretch Distance vs Delay Period (in Months)", 
        labels={"Stretch (Kms)": "Stretch Distance (Kms)", "Delay Period (Months)": "Delay Period (Months)"}
    )
    fig_scatter.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    fig_scatter.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        hovermode="closest",
        xaxis=dict(
            title="Stretch Distance (Kms)",
            tickmode='array',
            tickvals=[0, 100, 200, 300, 400, 500, 600],
            ticktext=['0', '100', '200', '300', '400', '500', '600']
        ),
        yaxis=dict(
            title="Delay Period (Months)",
            tickmode='array',
            tickvals=[-100, -50, 0, 50, 100, 150, 200],
            ticktext=['-100', '-50', '0', '50', '100', '150', '200']
        )
    )
    st.plotly_chart(fig_scatter)

    # Box plot of lane vs delay period
    st.write("**Lane Configuration vs Delay Period**")
    st.write("This box plot shows the distribution of delay periods for different lane configurations. It helps to identify if certain lane configurations are more prone to delays.")
    fig_box = px.box(
        filtered_df, 
        x="Lane", 
        y="Delay Period (Months)", 
        title="Lane Configuration vs Delay Period (in Months)", 
        labels={"Lane": "Lane Configuration", "Delay Period (Months)": "Delay Period (Months)"}
    )
    fig_box.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    fig_box.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        hovermode="closest",
        yaxis=dict(
            title="Delay Period (Months)",
            tickmode='array',
            tickvals=[-100, -50, 0, 50, 100, 150, 200],
            ticktext=['-100', '-50', '0', '50', '100', '150', '200']
        )
    )
    st.plotly_chart(fig_box)

    # Pie chart of mode distribution
    st.write("**Project Mode Distribution**")
    st.write("This pie chart shows the distribution of projects by mode. It helps to understand the proportion of different project modes in the dataset.")
    fig_pie = px.pie(filtered_df, names="Mode", title="Project Mode Distribution")
    fig_pie.update_traces(textinfo='percent+label')
    fig_pie.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )
    st.plotly_chart(fig_pie)

    # Prepare data for forecasting
    forecast_df = filtered_df[['Construction Commencement', 'Delay Period (Days)']].rename(columns={'Construction Commencement': 'ds', 'Delay Period (Days)': 'y'})

    # Fit the forecasting model
    model_forecast = Prophet()
    model_forecast.fit(forecast_df)

    # Make future dataframe
    future = model_forecast.make_future_dataframe(periods=365)
    forecast = model_forecast.predict(future)

    # Plot forecasting results
    st.write("**Year-Based Forecasting of Delay Period**")
    st.write("This line chart shows the predicted delay periods over time. It helps to understand the trend and forecast future delays.")
    fig_forecast = px.line(forecast, x='ds', y='yhat', title='Year-Based Forecasting of Delay Period (in Days)', labels={'ds': 'Date', 'yhat': 'Predicted Delay (Days)'})
    fig_forecast.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis_title="Date",
        yaxis_title="Predicted Delay (Days)"
    )
    st.plotly_chart(fig_forecast)

    # Create a choropleth map based on the number of projects in each state
    state_project_counts = df['State'].value_counts().reset_index()
    state_project_counts.columns = ['State', 'Project Count']

    # Merge the project counts with the original dataframe to include stretch names
    df_with_counts = df.merge(state_project_counts, on='State')

    # Load the Indian states GeoJSON file
    india_states_geojson_url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
    india_states_geojson = requests.get(india_states_geojson_url).json()

    # Map chart visualization
    st.write("### Project Locations on Map (India)")
    st.write("This map shows the number of projects by state. Hover over a state to see the stretch names.")
    map_chart = px.choropleth(
        df_with_counts,  # Use the merged dataframe
        geojson=india_states_geojson,  # Use the Indian states GeoJSON file
        locations="State",  # Column in your dataset with state names
        featureidkey="properties.ST_NM",  # Key in GeoJSON for state names
        color="Project Count",  # Column to determine color intensity
        hover_name="State",  # Column to display on hover
        title="Number of Projects by State in India",
        color_continuous_scale=px.colors.sequential.Plasma
    )
    map_chart.update_geos(fitbounds="locations", visible=False)  # Focus on India
    map_chart.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    st.plotly_chart(map_chart)

    # Leaderboard for Top Delayed Projects
    st.write("### Top 5 Most Delayed Projects")

    # Get the top 5 most delayed projects
    top_delayed = df.nlargest(5, 'Delay Period (Days)')[['Project Name', 'Delay Period (Days)']]

    # Reset the index and add a serial number column
    top_delayed = top_delayed.reset_index(drop=True)
    top_delayed.index = top_delayed.index + 1  # Start serial number from 1
    top_delayed = top_delayed.rename_axis('S.No.').reset_index()  # Add serial number as a column

    # Display the updated table without the default index
    st.table(top_delayed[['S.No.', 'Project Name', 'Delay Period (Days)']].set_index('S.No.'))
    # Calculate summary metrics
    total_projects = len(filtered_df)
    completed_projects = len(filtered_df[filtered_df['Project Status'] == 'Completed'])
    ongoing_projects = len(filtered_df[filtered_df['Project Status'] == 'Ongoing'])
    delayed_projects = len(filtered_df[filtered_df['Project Status'] == 'Delayed'])

    # Calculate construction period range
    construction_min = filtered_df['Construction Period (Months)'].min()
    construction_max = filtered_df['Construction Period (Months)'].max()

    # Calculate delay period range
    delay_min = filtered_df['Delay Period (Days)'].min()
    delay_max = filtered_df['Delay Period (Days)'].max()

    # Calculate stretch distance range
    stretch_min = filtered_df['Stretch (Kms)'].min()
    stretch_max = filtered_df['Stretch (Kms)'].max()
 # Generate summary text
    summary_text = f"""
    The dataset contains information on {total_projects} infrastructure projects, categorized as Completed, Ongoing, or Delayed.
    Among these projects, {completed_projects} have been completed, {ongoing_projects} are ongoing, and {delayed_projects} are delayed.
    The construction periods range from {construction_min} to {construction_max} months, with delay periods varying from {delay_min} to {delay_max} days.
    The stretch distances range from {stretch_min} to {stretch_max} kilometers.
    Notably, several ongoing and delayed projects exhibit significant delays, indicating potential challenges in timely completion.
    """
def generate_report(filtered_df, mse, prediction, scatter_df, delay_trend, comparison_df, top_delayed,
                    fig_status, fig_trend, fig_scatter, fig_line, fig_hist, fig_box, fig_pie, fig_forecast, map_chart,
                    construction_period, stretch_km, lane):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []

    # Custom style for headings
    heading_style = ParagraphStyle(
        name='Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.darkblue,
        spaceAfter=12,
    )

    # Title
    story.append(Paragraph("Project Delay Prediction Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Introduction
    story.append(Paragraph("This report provides an analysis of project delays based on the uploaded dataset. It includes key metrics, visualizations, and predictions.", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Input Details
    story.append(Paragraph("Input Details", heading_style))
    input_data = [
        ["Construction Period (Months)", construction_period],
        ["Stretch Distance (Kms)", stretch_km],
        ["Lane Configuration", lane]
    ]
    input_table = Table(input_data, colWidths=[200, 100])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(input_table)
    story.append(Spacer(1, 12))

    # Key Metrics
    story.append(Paragraph("Key Metrics", heading_style))
    story.append(Paragraph(f"Mean Squared Error (MSE): {mse:,.2f}", styles['BodyText']))
    story.append(Paragraph(f"Predicted Delay: {prediction:,.0f} days", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Prediction Details
    story.append(Paragraph("Prediction Details", heading_style))
    
    # Convert days to months and years for better understanding
    prediction_months = prediction / 30
    prediction_years = prediction / 365

    # Calculate the expected completion date
    loa_date = pd.to_datetime('today')  # Assuming today's date as LOA date for prediction
    expected_completion_date = loa_date + pd.DateOffset(months=construction_period) + pd.DateOffset(days=prediction)

    # Add prediction details to the report
    if prediction < 0:
        story.append(Paragraph(f"The project is predicted to be completed **{abs(prediction):,.0f} days** earlier than scheduled.", styles['BodyText']))
        story.append(Paragraph(f"This is approximately **{abs(prediction_months):,.1f} months** or **{abs(prediction_years):,.1f} years** earlier.", styles['BodyText']))
    else:
        story.append(Paragraph(f"The project is predicted to be delayed by **{prediction:,.0f} days**.", styles['BodyText']))
        story.append(Paragraph(f"This is approximately **{prediction_months:,.1f} months** or **{prediction_years:,.1f} years**.", styles['BodyText']))
    
    story.append(Paragraph(f"**Expected Completion Date:** {expected_completion_date.date()}", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Generate Summary
    story.append(Paragraph("Project Completion Prediction Summary", heading_style))
    
    # Calculate summary metrics
    total_projects = len(filtered_df)
    completed_projects = len(filtered_df[filtered_df['Project Status'] == 'Completed'])
    ongoing_projects = len(filtered_df[filtered_df['Project Status'] == 'Ongoing'])
    delayed_projects = len(filtered_df[filtered_df['Project Status'] == 'Delayed'])
    
    construction_min = filtered_df['Construction Period (Months)'].min()
    construction_max = filtered_df['Construction Period (Months)'].max()
    
    delay_min = filtered_df['Delay Period (Days)'].min()
    delay_max = filtered_df['Delay Period (Days)'].max()
    
    stretch_min = filtered_df['Stretch (Kms)'].min()
    stretch_max = filtered_df['Stretch (Kms)'].max()
    
   
    
    story.append(Paragraph(summary_text, styles['BodyText']))
    story.append(Spacer(1, 12))

    # Add a page break to start the next section on a new page
    story.append(PageBreak())

    # Function to add images with proper alignment
    def add_image_to_story(fig, title):
        # Update layout for better text rendering
        fig.update_layout(
            font=dict(family="Arial", size=12, color="black"),  # Use a standard font
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Export the figure with high resolution
        img_buffer = io.BytesIO()
        fig.write_image(img_buffer, format='png', scale=3, engine='kaleido')  # High-quality export
        img_buffer.seek(0)

        # Add the image to the PDF story
        story.append(Paragraph(title, heading_style))
        img = Image(img_buffer, width=500, height=300)  # Adjust size for alignment
        img.hAlign = 'CENTER'  # Center-align the image
        story.append(img)
        story.append(Spacer(1, 12))  # Add spacing after the image
        story.append(PageBreak())  # Add a page break after each image

    # Add all images to the report
    add_image_to_story(fig_status, "Project Status Distribution")
    add_image_to_story(fig_trend, "Average Delay Trend Over Time")
    add_image_to_story(fig_scatter, "Actual vs Predicted Delay Period")
    add_image_to_story(fig_line, "Actual vs Predicted Delay Period Over Time")
    add_image_to_story(fig_hist, "Distribution of Delay Periods")
    add_image_to_story(fig_box, "Lane Configuration vs Delay Period")
    add_image_to_story(fig_pie, "Project Mode Distribution")
    add_image_to_story(fig_forecast, "Year-Based Forecasting of Delay Period")
    add_image_to_story(map_chart, "Project Locations on Map (India)")

    # Top 5 Most Delayed Projects
    story.append(Paragraph("Top 5 Most Delayed Projects", heading_style))
    top_delayed_data = [['No.', 'Project Name', 'Delay Period (Days)']] + top_delayed.values.tolist()

    # Define the table with adjusted column widths and dynamic row heights
    table = Table(top_delayed_data, colWidths=[50, 500, 100])

    # Apply table styles
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),  # Header row background
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Header row text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center-align all cells
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Header row font
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Header row padding
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),  # Data rows background
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Add grid lines
        ('WORDWRAP', (1, 0), (1, -1), True),  # Wrap text in the "Project Name" column
    ]))

    # Add the table to the story
    story.append(table)
    story.append(Spacer(1, 12))

    # Build the PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# Check if prediction has been made and show the Generate Report button
if 'prediction_made' in st.session_state and st.session_state['prediction_made']:
    if st.button("Generate Report"):
        # Ensure prediction is defined
        if 'prediction' not in st.session_state:
            st.session_state['prediction'] = 0  # Default value if no prediction is made

        # Generate the report
        report_buffer = generate_report(
            filtered_df, mse, st.session_state['prediction'], scatter_df, delay_trend, comparison_df, top_delayed,
            fig_status, fig_trend, fig_scatter, fig_line, fig_hist, fig_box, fig_pie, fig_forecast, map_chart,
            construction_period, stretch_km, lane
        )

        # Display a link to view the report
        base64_pdf = base64.b64encode(report_buffer.getvalue()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

        # Download button for the report
        st.download_button(
            label="Download Report",
            data=report_buffer,
            file_name="project_delay_report.pdf",
            mime="application/pdf"
        )
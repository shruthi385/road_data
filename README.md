# road_data
# Project Delay Prediction Dashboard

## Introduction
The Project Delay Prediction Dashboard is a comprehensive tool designed to help project managers and stakeholders predict delays in construction projects. By uploading project data in CSV or Excel format, users can leverage machine learning models to forecast potential delays and gain insights into various project parameters. The dashboard offers a user-friendly interface with interactive visualizations, allowing users to explore and analyze project data effectively.

## Technical Summary

### Key Features
- **Data Upload**: Supports CSV and Excel file formats for project data input.
- **Data Preview**: Provides an initial preview of the uploaded dataset.
- **Data Handling**: Automatically handles missing values using forward fill.
- **Feature Engineering**: Calculates important features such as construction period, time taken for completion, scheduled time for completion, and delay period.
- **Data Filtering**: Allows users to filter data based on project commencement year, lane configuration, project mode, construction duration, concession period, and stretch distance.
- **Model Training**: Utilizes a RandomForestRegressor model to predict project delays.
- **Model Evaluation**: Displays mean squared error of the model predictions.
- **Interactive Visualizations**: Includes scatter plots, histograms, box plots, bar charts, and pie charts for data visualization.
- **Forecasting**: Uses the Prophet model for year-based forecasting of delay periods.

### Technologies Used

- **Streamlit**: Provides the framework for building the interactive web application.
- **Pandas**: Used for data manipulation and preprocessing.
- **Scikit-learn**: Implements the machine learning model (RandomForestRegressor) and handles train-test splitting.
- **Plotly**: Creates interactive visualizations including scatter plots, histograms, box plots, bar charts, and pie charts.
- **Prophet**: Performs time series forecasting to predict future delay periods.
- **FontAwesome**: Adds icons to enhance the user interface.

### Detailed Workflow

1. **Data Upload and Preview**:
   - Users can upload a CSV or Excel file containing project data.
   - The dashboard provides a preview of the uploaded dataset.

2. **Data Handling and Feature Engineering**:
   - Missing values are handled using forward fill.
   - Key features such as construction period, time taken for completion, scheduled time for completion, and delay period are calculated.

3. **Data Filtering**:
   - Users can apply filters to the dataset based on various project parameters.
   - Filtered data is displayed for further analysis.

4. **Model Training and Evaluation**:
   - A RandomForestRegressor model is trained on the filtered dataset.
   - The model's performance is evaluated using mean squared error.

5. **Prediction and Visualization**:
   - Users can input new project parameters to predict potential delays.
   - The dashboard provides interactive visualizations to compare actual vs. predicted delays and explore various data distributions.

6. **Forecasting**:
   - The Prophet model is used to forecast future delay periods based on historical data.
   - Forecasting results are displayed as an interactive line chart.

### Conclusion
The Project Delay Prediction Dashboard is a powerful tool that combines machine learning and interactive visualizations to help project managers predict and analyze construction delays. By leveraging advanced technologies and user-friendly features, the dashboard provides valuable insights into project performance and potential risks.

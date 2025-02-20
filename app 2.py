import streamlit as st
import pandas as pd

# User input for name
user_name = st.text_input("Please enter city name:", "")

# Title and description with personalization
if user_name:
    st.title(f"Welcome  to the Data Analysis App!")
else:
    st.title("Welcome to the Data Analysis App!")
st.write("This app allows you to explore and analyze data from the uploaded CSV file.")

# Load the CSV file
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# File path for the uploaded CSV file
file_path = '../anoushka/modified_file2.csv'  # Path to the uploaded file

# Only show data analysis if name is provided
if user_name:
    # Load and display data
    data = load_data(file_path)
    if data is not None:
        st.write(f"### Hello {user_name}, here's your Data Preview:")
        st.dataframe(data)

        # Show basic statistics
        st.write(f"### {user_name}'s Basic Statistics:")
        st.write(data.describe())

        # Filter data
        columns = list(data.columns)
        st.write("### Filter Your Data:")
        filter_column = st.selectbox("Select a column to filter:", columns)
        if filter_column:
            unique_values = data[filter_column].unique()
            filter_value = st.selectbox(f"Select a value for {filter_column}:", unique_values)
            filtered_data = data[data[filter_column] == filter_value]
            st.write("Filtered Data:")
            st.dataframe(filtered_data)

        # Visualize data
        st.write("### Your Data Visualization:")
        chart_column = st.selectbox("Select a column for visualization:", columns)
        if chart_column:
            st.bar_chart(data[chart_column].value_counts())
    else:
        st.write("No data available to display. Please check the file path or format.")

    # Add a personalized note
    st.write(f"\nThank you for using this app, {user_name}! Let me know if you need any help!")
else:
    st.write("Please enter your name to start exploring the data!")

# Add a note about the notebook integration (future functionality)
st.write("\nThis is a basic Streamlit app. Integration with the uploaded Jupyter notebook is not yet implemented. Let me know how you'd like to extend this functionality!")

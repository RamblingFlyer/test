import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin

# Set page config
st.set_page_config(page_title="Water Level Analysis Dashboard", layout="wide")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ['Home', 'Data Analysis', 'Predictions', 'Visualization'])

# Data loading classes
class DataLoader:
    @staticmethod
    def load_rainfall_data(file):
        try:
            rainfall_df = pd.read_csv(file)
            rainfall_features = rainfall_df.groupby('YEAR').agg({
                'ANNUAL': 'mean',
                'June-September': 'mean',
                'Mar-May': 'mean',
                'Jan-Feb': 'mean',
                'Oct-Dec': 'mean'
            }).reset_index()
            return rainfall_features
        except Exception as e:
            st.error(f"Error loading rainfall data: {e}")
            return None

    @staticmethod
    def load_canal_data(file):
        try:
            canal_df = pd.read_csv(file)
            canal_stats = pd.DataFrame({
                'Canal_Flow_Mean': [canal_df['Canal Flow'].mean()],
                'Canal_Flow_Std': [canal_df['Canal Flow'].std()],
                'Soil_Moisture_Mean': [canal_df['Soil Moisture'].mean()],
                'Aquifer_Thickness_Mean': [canal_df['Aquifer Thickness'].mean()],
                'Hydraulic_Conductivity_Mean': [canal_df['Hydraulic Conductivity'].mean()]
            })
            return canal_stats
        except Exception as e:
            st.error(f"Error loading canal data: {e}")
            return None

    @staticmethod
    def load_water_level_data(file):
        try:
            water_level_df = pd.read_excel(file)
            processed_data = []

            for year in range(2016, 2024):
                if year == 2022:
                    pre_col = f'Pre_{year}'
                    post_col = 'Post_2022'
                elif year == 2023:
                    pre_col = f'Pre_{year}'
                    post_col = 'Post_2023'
                else:
                    pre_col = f'Pre_{year}'
                    post_col = f'Pst_{year}'

                if pre_col not in water_level_df.columns or post_col not in water_level_df.columns:
                    continue

                year_data = water_level_df[[
                    'Latitude', 'Longitude', 'Well_Depth',
                    pre_col, post_col
                ]].copy()

                year_data[pre_col] = pd.to_numeric(year_data[pre_col].replace(' ', np.nan), errors='coerce')
                year_data[post_col] = pd.to_numeric(year_data[post_col].replace(' ', np.nan), errors='coerce')

                year_data['Year'] = year
                year_data['Water_Level'] = (year_data[pre_col] + year_data[post_col]) / 2

                year_data = year_data.dropna(subset=['Water_Level'])

                if len(year_data) > 0:
                    year_data = year_data.drop([pre_col, post_col], axis=1)
                    processed_data.append(year_data)

            if processed_data:
                final_df = pd.concat(processed_data, ignore_index=True)
                return final_df
            else:
                st.error("No valid data processed")
                return None

        except Exception as e:
            st.error(f"Error loading water level data: {e}")
            return None

# Model class
class ImprovedStackedEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, rf_params=None, xgb_params=None):
        self.rf_params = rf_params if rf_params else {'random_state': 42}
        self.xgb_params = xgb_params if xgb_params else {'random_state': 42}
        self.rf_model = RandomForestRegressor(**self.rf_params)
        self.xgb_model = XGBRegressor(**self.xgb_params)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.rf_model.fit(X, y)
        rf_predictions = self.rf_model.predict(X)
        xgb_features = np.column_stack([X, rf_predictions])
        self.xgb_model.fit(xgb_features, y)
        return self

    def predict(self, X):
        rf_predictions = self.rf_model.predict(X)
        xgb_features = np.column_stack([X, rf_predictions])
        xgb_predictions = self.xgb_model.predict(xgb_features)
        return rf_predictions

    def predict_future(self, X_last, years=5):
        future_predictions = []
        current_data = X_last.copy()

        for year in range(years):
            pred = self.predict(current_data)
            future_predictions.append(pred[-1])  # Only append the last prediction
            current_data = np.roll(current_data, -1, axis=0)
            current_data[-1, 0] = 2024 + year + 1
            current_data[-1, 3:] = current_data[-2, 3:]
            current_data[-1, 6:11] = current_data[-2, 6:11]
            current_data[-1, 11:] = current_data[-2, 11:]
            current_data[-1, -2] = pred[-1]
        return future_predictions

# Home page
def show_home():
    st.title("Water Level Analysis Dashboard")
    st.write("""
    Welcome to the comprehensive Water Level Analysis Dashboard. This application provides:
    - Data analysis and visualization of water levels
    - Predictive modeling using machine learning
    - Interactive data exploration
    - Future water level predictions
    """)

# Data Analysis page
def show_data_analysis():
    st.title("Data Analysis")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(data)

        # Basic statistics
        st.write("### Basic Statistics:")
        st.write(data.describe())

        # Data filtering
        st.write("### Filter Data:")
        columns = list(data.columns)
        filter_column = st.selectbox("Select column to filter:", columns)
        if filter_column:
            unique_values = data[filter_column].unique()
            filter_value = st.selectbox(f"Select value for {filter_column}:", unique_values)
            filtered_data = data[data[filter_column] == filter_value]
            st.write("Filtered Data:")
            st.dataframe(filtered_data)

# Predictions page
def show_predictions():
    st.title("Water Level Predictions")

    # File uploaders
    rainfall_file = st.file_uploader("Upload Rainfall Data (CSV)", type=['csv'])
    water_level_file = st.file_uploader("Upload Water Level Data (Excel)", type=['xlsx'])
    canal_file = st.file_uploader("Upload Canal Data (CSV)", type=['csv'])

    if rainfall_file and water_level_file and canal_file:
        data_loader = DataLoader()
        
        # Load data
        rainfall_data = data_loader.load_rainfall_data(rainfall_file)
        canal_data = data_loader.load_canal_data(canal_file)
        water_level_data = data_loader.load_water_level_data(water_level_file)

        if all([rainfall_data is not None, canal_data is not None, water_level_data is not None]):
            # Preprocess data
            X = pd.merge(
                water_level_data,
                rainfall_data[['YEAR', 'ANNUAL', 'June-September', 'Mar-May', 'Jan-Feb', 'Oct-Dec']],
                left_on='Year',
                right_on='YEAR',
                how='left'
            )

            for col in canal_data.columns:
                X[col] = canal_data[col].iloc[0]

            y = X['Water_Level']
            X = X.drop(['Water_Level', 'YEAR'], axis=1)

            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = ImprovedStackedEnsemble()
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)
            future_predictions = model.predict_future(X_test[-3:], years=5)

            # Display results
            st.write("### Model Performance")
            st.write(f"R² Score: {r2_score(y_test, predictions):.4f}")
            st.write(f"MSE: {mean_squared_error(y_test, predictions):.4f}")

            # Plot predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_test, name='Actual'))
            fig.add_trace(go.Scatter(y=predictions, name='Predicted'))
            st.plotly_chart(fig)

            # Display future predictions
            st.write("### Future Predictions")
            years = range(2024, 2029)
            future_df = pd.DataFrame({
                'Year': years,
                'Predicted_Water_Level': future_predictions
            })
            st.write(future_df)

# Visualization page
def show_visualization():
    st.title("Data Visualization")
    uploaded_file = st.file_uploader("Upload Data for Visualization", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Select columns for visualization
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        # Scatter plot
        st.write("### Scatter Plot")
        x_axis = st.selectbox("X-axis", numeric_columns)
        y_axis = st.selectbox("Y-axis", numeric_columns)
        
        fig = px.scatter(data, x=x_axis, y=y_axis)
        st.plotly_chart(fig)
        
        # Line plot
        st.write("### Line Plot")
        time_column = st.selectbox("Time Column", data.columns)
        value_column = st.selectbox("Value Column", numeric_columns)
        
        fig = px.line(data, x=time_column, y=value_column)
        st.plotly_chart(fig)

# Main app logic
if page == 'Home':
    show_home()
elif page == 'Data Analysis':
    show_data_analysis()
elif page == 'Predictions':
    show_predictions()
else:
    show_visualization()
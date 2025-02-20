import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Water Level Prediction App", layout="wide")

# Title and description
st.title("Water Level Prediction Dashboard")
st.write("This application helps analyze and predict water levels based on rainfall data.")

# Create sidebar for data upload and model parameters
st.sidebar.header("Data Input")

# File upload section
rainfall_file = st.sidebar.file_uploader("Upload Rainfall Data (CSV)", type=['csv'])
water_level_file = st.sidebar.file_uploader("Upload Water Level Data (Excel)", type=['xlsx'])
canal_file = st.sidebar.file_uploader("Upload Canal Data (CSV)", type=['csv'])

class DataLoader:
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
                raise ValueError("No valid data processed")

        except Exception as e:
            st.error(f"Error loading water level data: {e}")
            return None

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess_data(self, rainfall_df, water_level_df, canal_df=None):
        try:
            if rainfall_df is None or water_level_df is None:
                raise ValueError("One or more input DataFrames are None")

            rainfall_df = rainfall_df.reset_index(drop=True)
            water_level_df = water_level_df.reset_index(drop=True)
            rainfall_df['YEAR'] = rainfall_df['YEAR'].astype(int)
            water_level_df['Year'] = water_level_df['Year'].astype(int)

            merged_data = pd.merge(
                water_level_df,
                rainfall_df[['YEAR', 'ANNUAL', 'June-September', 'Mar-May', 'Jan-Feb', 'Oct-Dec']],
                left_on='Year',
                right_on='YEAR',
                how='left'
            )

            if canal_df is not None:
                for col in canal_df.columns:
                    merged_data[col] = canal_df[col].iloc[0]

            merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
            merged_data['MonthOfYear'] = 6
            merged_data['DayOfYear'] = 182

            feature_columns = [
                'Year', 'MonthOfYear', 'DayOfYear',
                'Latitude', 'Longitude', 'Well_Depth',
                'ANNUAL', 'June-September', 'Mar-May',
                'Jan-Feb', 'Oct-Dec'
            ]

            if canal_df is not None:
                feature_columns.extend(['Canal_Flow_Mean', 'Canal_Flow_Std', 'Soil_Moisture_Mean',
                                      'Aquifer_Thickness_Mean', 'Hydraulic_Conductivity_Mean'])

            existing_columns = [col for col in feature_columns if col in merged_data.columns]

            X = merged_data[existing_columns].copy()
            y = merged_data['Water_Level'].copy()

            return X, y

        except Exception as e:
            st.error(f"Error in data preprocessing: {e}")
            return None, None

class ImprovedStackedEnsemble:
    def __init__(self):
        self.rf_model = RandomForestRegressor(random_state=42)
        self.xgb_model = XGBRegressor(random_state=42)
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
        return self.xgb_model.predict(xgb_features)

    def predict_future(self, X_last, years=5):
        future_predictions = []
        current_data = X_last.copy()

        for year in range(years):
            pred = self.predict(current_data)
            future_predictions.append(pred.mean())  # Take mean of predictions for each year
            current_data = np.roll(current_data, -1, axis=0)
            current_data[-1, 0] = 2024 + year + 1
            current_data[-1, 3:] = current_data[-2, 3:]
            current_data[-1, 6:11] = current_data[-2, 6:11]

        return future_predictions

# Main application logic
if rainfall_file and water_level_file:
    # Load data
    data_loader = DataLoader()
    rainfall_data = data_loader.load_rainfall_data(rainfall_file)
    water_level_data = data_loader.load_water_level_data(water_level_file)
    canal_data = data_loader.load_canal_data(canal_file) if canal_file else None

    if rainfall_data is not None and water_level_data is not None:
        # Preprocess data
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess_data(rainfall_data, water_level_data, canal_data)

        if X is not None and y is not None:
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Spatial Analysis", "Canal Analysis", "Predictions"])

            with tab1:
                st.header("Data Overview")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Rainfall Trends")
                    fig_rainfall = px.line(rainfall_data, x='YEAR', y=['ANNUAL', 'June-September', 'Mar-May', 'Jan-Feb', 'Oct-Dec'],
                                          title='Detailed Rainfall Trends Over Years')
                    st.plotly_chart(fig_rainfall)

                with col2:
                    st.subheader("Water Level Distribution")
                    fig_water = px.box(water_level_data, x='Year', y='Water_Level')
                    st.plotly_chart(fig_water)

            with tab2:
                st.header("Spatial Distribution of Water Levels")
                year_filter = st.selectbox("Select Year", sorted(water_level_data['Year'].unique()))
                filtered_data = water_level_data[water_level_data['Year'] == year_filter]

                fig_map = px.scatter_mapbox(filtered_data,
                                          lat='Latitude',
                                          lon='Longitude',
                                          color='Water_Level',
                                          size='Well_Depth',
                                          hover_data=['Water_Level', 'Well_Depth'],
                                          zoom=6,
                                          title=f'Water Levels in {year_filter}')
                fig_map.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig_map)

            with tab3:
                st.header("Canal Analysis")
                if canal_data is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Canal Flow Statistics")
                        st.write(pd.DataFrame({
                            'Metric': ['Mean Flow', 'Flow Std Dev', 'Mean Soil Moisture',
                                      'Mean Aquifer Thickness', 'Mean Hydraulic Conductivity'],
                            'Value': [canal_data['Canal_Flow_Mean'].iloc[0],
                                     canal_data['Canal_Flow_Std'].iloc[0],
                                     canal_data['Soil_Moisture_Mean'].iloc[0],
                                     canal_data['Aquifer_Thickness_Mean'].iloc[0],
                                     canal_data['Hydraulic_Conductivity_Mean'].iloc[0]]
                        }))
                    with col2:
                        st.subheader("Canal Flow Distribution")
                        fig_canal = go.Figure(data=[go.Bar(
                            x=['Canal Flow', 'Soil Moisture', 'Aquifer Thickness', 'Hydraulic Conductivity'],
                            y=[canal_data['Canal_Flow_Mean'].iloc[0],
                               canal_data['Soil_Moisture_Mean'].iloc[0],
                               canal_data['Aquifer_Thickness_Mean'].iloc[0],
                               canal_data['Hydraulic_Conductivity_Mean'].iloc[0]]
                        )])
                        st.plotly_chart(fig_canal)
                else:
                    st.info("Please upload Canal data to view analysis")

            with tab4:
                st.header("Water Level Predictions")
                if st.button("Train Model and Generate Predictions"):
                    with st.spinner("Training model and generating predictions..."):
                        # Train model
                        model = ImprovedStackedEnsemble()
                        model.fit(X, y)

                        # Generate future predictions
                        future_years = 5
                        future_preds = model.predict_future(X.values[-3:], years=future_years)

                        # Create prediction plot
                        years = list(range(2024, 2024 + future_years))
                        pred_df = pd.DataFrame({
                            'Year': years,
                            'Predicted_Water_Level': future_preds
                        })

                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(x=water_level_data['Year'].unique(),
                                                     y=water_level_data.groupby('Year')['Water_Level'].mean(),
                                                     name='Historical Data'))
                        fig_pred.add_trace(go.Scatter(x=pred_df['Year'],
                                                     y=pred_df['Predicted_Water_Level'],
                                                     name='Predictions',
                                                     line=dict(dash='dash')))
                        fig_pred.update_layout(title='Water Level Predictions',
                                              xaxis_title='Year',
                                              yaxis_title='Water Level')
                        st.plotly_chart(fig_pred)

                        # Display prediction table
                        st.subheader("Predicted Water Levels")
                        st.dataframe(pred_df)

else:
    st.info("Please upload both Rainfall and Water Level data files to begin analysis.")
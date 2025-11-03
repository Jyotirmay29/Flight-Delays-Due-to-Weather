# flight_delay_predictor.py
#
# A complete, end-to-end script to build a machine learning model for
# predicting flight delays based on real data from CSV files.
#
# This script covers:
# 1.  Data Loading & Merging: Loads flight and weather data and merges them.
# 2.  Data Cleaning & Preprocessing: Handles missing values and defines the target.
# 3.  Exploratory Data Analysis (EDA) & Visualization: Explores the data.
# 4.  Feature Engineering: Creates new features to improve model performance.
# 5.  Model Training: Builds and trains a RandomForestClassifier.
# 6.  Model Evaluation: Assesses the model's performance.
# 7.  Feature Importance Analysis: Identifies key predictors.
# 8.  Prediction on New Data: Shows how to make actionable predictions.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Loading and Merging ---
def load_and_merge_data(flight_csv_path, weather_csv_path):
    """Loads flight and weather data from CSV files and merges them."""
    print("Step 1: Loading and merging data...")
    try:
        flights_df = pd.read_csv(flight_csv_path)
        weather_df = pd.read_csv(weather_csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure '{flight_csv_path}' and '{weather_csv_path}' are in the same directory.")
        return None

    # For merging, we need a common key. Let's use airport and date.
    # Convert date columns to datetime objects for proper merging.
    flights_df['DATE'] = pd.to_datetime(flights_df[['YEAR', 'MONTH', 'DAY']])
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
    
    # Merge weather data based on origin airport and date
    merged_df = pd.merge(flights_df, weather_df, left_on=['ORIGIN_AIRPORT', 'DATE'], right_on=['AIRPORT', 'DATE'], how='left')
    
    print("Data loading and merging complete.\n")
    return merged_df

# --- 2. Data Cleaning and Preprocessing ---
def clean_and_prepare_data(df):
    """Cleans the merged dataframe and prepares the target variable."""
    print("Step 2: Cleaning data and preparing target variable...")
    # Handle potential missing values after the merge (e.g., weather data not found)
    # For simplicity, we'll fill numerical weather data with the median
    for col in ['TEMPERATURE_CELSIUS', 'WIND_SPEED_KMH', 'PRECIPITATION_MM', 'VISIBILITY_KM']:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            
    # Define our target variable: A flight is 'delayed' if arrival_delay > 15 minutes.
    # This is a common industry standard.
    df['DELAY_STATUS'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
    
    # Drop columns that are not needed for the model or have been processed
    df.drop(columns=['YEAR', 'MONTH', 'DAY', 'DATE', 'AIRPORT', 'ARRIVAL_DELAY', 'DEPARTURE_DELAY'], inplace=True)
    df.dropna(inplace=True) # Drop any remaining rows with missing data
    
    print("Data cleaning complete.\n")
    return df

# --- 3. Exploratory Data Analysis (EDA) ---
def perform_eda(df):
    """Performs and visualizes basic EDA on the real dataset."""
    print("Step 3: Performing Exploratory Data Analysis...")
    print("Dataset Information:")
    df.info()
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nDelay Status Distribution:")
    print(df['DELAY_STATUS'].value_counts(normalize=True))

    # Visualize the average delay by airline
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df.groupby('AIRLINE')['DELAY_STATUS'].mean().sort_values(ascending=False).index,
                y=df.groupby('AIRLINE')['DELAY_STATUS'].mean().sort_values(ascending=False).values)
    plt.title('Proportion of Delayed Flights by Airline')
    plt.xlabel('Airline')
    plt.ylabel('Proportion Delayed')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("delays_by_airline.png")
    print("\nEDA complete. Plot saved to 'delays_by_airline.png'.\n")
    plt.close()

# --- 4. Feature Engineering ---
def engineer_features(df):
    """Creates new features to potentially improve model performance."""
    print("Step 4: Performing Feature Engineering...")
    # Extract time-based features from 'SCHEDULED_DEPARTURE' (format HHMM)
    df['departure_hour'] = df['SCHEDULED_DEPARTURE'] // 100
    df['is_weekend'] = (df['DAY_OF_WEEK'] >= 6).astype(int)
    
    print("Feature engineering complete. Added 'departure_hour' and 'is_weekend'.\n")
    return df

# --- Main Execution ---
if __name__ == "__main__":
    # Define file paths for the datasets
    FLIGHTS_CSV = 'flights.csv'
    WEATHER_CSV = 'weather.csv'

    # Load, merge, clean, and engineer features
    flight_data = load_and_merge_data(FLIGHTS_CSV, WEATHER_CSV)
    
    if flight_data is not None:
        flight_data = clean_and_prepare_data(flight_data)
        perform_eda(flight_data)
        flight_data = engineer_features(flight_data)

        # Define features (X) and target (y)
        X = flight_data.drop('DELAY_STATUS', axis=1)
        y = flight_data['DELAY_STATUS']

        # --- 5. Data Preprocessing Setup ---
        print("Step 5: Setting up data preprocessing pipeline...")
        categorical_features = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
        numerical_features = ['DAY_OF_WEEK', 'SCHEDULED_DEPARTURE', 'DISTANCE', 
                              'TEMPERATURE_CELSIUS', 'WIND_SPEED_KMH', 'PRECIPITATION_MM', 
                              'VISIBILITY_KM', 'departure_hour', 'is_weekend']
        
        # Drop features that are not in the dataframe to avoid errors
        X_cols = X.columns
        categorical_features = [f for f in categorical_features if f in X_cols]
        numerical_features = [f for f in numerical_features if f in X_cols]


        # Create preprocessing pipelines for numerical and categorical data
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        print("Preprocessing pipeline is ready.\n")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # --- 6. Model Training ---
        print("Step 6: Training the machine learning model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('classifier', model)])

        model_pipeline.fit(X_train, y_train)
        print("Model training complete.\n")

        # --- 7. Model Evaluation ---
        print("Step 7: Evaluating the model...")
        y_pred = model_pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['On-Time', 'Delayed']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Model evaluation complete.\n")
        
        # --- 8. Feature Importance Analysis ---
        print("Step 8: Analyzing feature importances...")
        try:
            ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
            all_feature_names = numerical_features + list(ohe_feature_names)
            
            importances = model_pipeline.named_steps['classifier'].feature_importances_
            
            feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(15)
            
            print("Top 15 Most Important Features:")
            print(feature_importance_df)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_importance_df, x='importance', y='feature')
            plt.title('Top 15 Feature Importances for Predicting Flight Delays')
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig("feature_importances.png")
            print("\nFeature importance analysis complete. Plot saved to 'feature_importances.png'.\n")
            plt.close()
        except Exception as e:
            print(f"Could not perform feature importance analysis. Error: {e}\n")


        # --- 9. Prediction on New Data ---
        # This section is commented out as it requires manual data entry matching the new format.
        # To use it, create a DataFrame with the same columns as the training data.
        print("Step 9: Prediction on new data example is available in the script comments.")
        # new_flights = pd.DataFrame({
        #     'DAY_OF_WEEK': [5], # Friday
        #     'AIRLINE': ['DL'], # Delta
        #     'ORIGIN_AIRPORT': ['JFK'],
        #     'DESTINATION_AIRPORT': ['LAX'],
        #     'SCHEDULED_DEPARTURE': [1800], # 6:00 PM
        #     'DISTANCE': [2475],
        #     'TEMPERATURE_CELSIUS': [-5],
        #     'WIND_SPEED_KMH': [85],
        #     'PRECIPITATION_MM': [10],
        #     'VISIBILITY_KM': [1]
        # })
        # new_flights = engineer_features(new_flights)
        # predictions = model_pipeline.predict(new_flights)
        # print(f"Prediction for new flight: {'Delayed' if predictions[0] == 1 else 'On-Time'}")

        print("\nScript finished successfully.")

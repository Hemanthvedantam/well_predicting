from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime, timedelta
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import threading
import warnings
warnings.filterwarnings('ignore')
import openpyxl

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Global variables for data and models
data_lock = threading.Lock()
last_data_load_time = None
df_data = None
all_assets = []
asset_well_map = {}
well_asset_map = {}
ml_model_optimal = None
ml_model_failure = None
scaler_optimal = None
scaler_failure = None

def load_and_process_data():
    """Load and process the CSV data with missing value handling"""
    global df_data, all_assets, asset_well_map, well_asset_map

    csv_path = os.path.join(os.path.dirname(__file__), 'well_data.xlsx')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The data file '{csv_path}' was not found.")

    df_data = pd.read_excel(csv_path, engine='openpyxl')
    df_data.columns = df_data.columns.str.strip()
    df_data = handle_missing_values(df_data)

    all_assets = df_data['FieldInstallationName'].unique().tolist()
    asset_well_map = {}
    well_asset_map = {}

    for asset in all_assets:
        wells = df_data[df_data['FieldInstallationName'] == asset]['WellName'].unique().tolist()
        asset_well_map[asset] = wells
        for well in wells:
            well_asset_map[well] = asset

def handle_missing_values(df):
    df['WorkCompletedOn'] = pd.to_datetime(df['WorkCompletedOn'], errors='coerce')
    df['WorkCompletedOn'].fillna(pd.Timestamp.now(), inplace=True)
    df['Maintained'].fillna('FALSE', inplace=True)
    df['ManualOverride'].fillna('FALSE', inplace=True)
    df['TestResult'].fillna('P', inplace=True)
    df['TestTypeCode'].fillna('Standard', inplace=True)
    df['StartPressure'].fillna(df['StartPressure'].median(), inplace=True)
    df['FinishPressure'].fillna(df['FinishPressure'].median(), inplace=True)
    return df

def calculate_optimal_interval(intervals, failure_rate):
    if not intervals or len(intervals) == 0:
        return 90
    avg_interval = np.mean(intervals)
    safety_factor = 0.9
    if failure_rate > 20:
        safety_factor = 0.6
    elif failure_rate > 10:
        safety_factor = 0.75
    return max(30, min(365, avg_interval * safety_factor))

def train_ml_models():
    global ml_model_optimal, ml_model_failure, scaler_optimal, scaler_failure

    if df_data is None or len(df_data) == 0:
        return

    features_data = []
    target_optimal = []
    target_failure = []

    for asset in all_assets:
        asset_data = df_data[df_data['FieldInstallationName'] == asset]

        for well in asset_well_map[asset]:
            well_data = asset_data[asset_data['WellName'] == well].copy()
            if len(well_data) < 2:
                continue

            well_data = well_data.sort_values('WorkCompletedOn')
            total_tests = len(well_data)
            failed_tests = len(well_data[well_data['TestResult'] != 'P'])
            failure_rate = (failed_tests / total_tests) * 100

            dates = well_data['WorkCompletedOn'].tolist()
            intervals = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]

            if intervals:
                avg_interval = np.mean(intervals)
                max_interval = max(intervals)
                min_interval = min(intervals)
                std_interval = np.std(intervals)
                avg_start_pressure = well_data['StartPressure'].mean()
                avg_finish_pressure = well_data['FinishPressure'].mean()
                pressure_variance = well_data['StartPressure'].var()

                features = [
                    total_tests,
                    failure_rate,
                    avg_interval,
                    max_interval,
                    min_interval,
                    std_interval,
                    avg_start_pressure,
                    avg_finish_pressure,
                    pressure_variance
                ]

                optimal_interval = calculate_optimal_interval(intervals, failure_rate)
                features_data.append(features)
                target_optimal.append(optimal_interval)
                target_failure.append(failure_rate)

    if len(features_data) > 5:
        X = np.array(features_data)
        y_optimal = np.array(target_optimal)
        y_failure = np.array(target_failure)

        scaler_optimal = StandardScaler()
        scaler_failure = StandardScaler()
        X_scaled_optimal = scaler_optimal.fit_transform(X)
        X_scaled_failure = scaler_failure.fit_transform(X)

        ml_model_optimal = RandomForestRegressor(n_estimators=100, random_state=42)
        ml_model_failure = RandomForestRegressor(n_estimators=100, random_state=42)

        ml_model_optimal.fit(X_scaled_optimal, y_optimal)
        ml_model_failure.fit(X_scaled_failure, y_failure)

def get_asset_overview(asset_name):
    asset_data = df_data[df_data['FieldInstallationName'] == asset_name]
    if asset_data.empty:
        return None

    wells = asset_well_map.get(asset_name, [])
    total_tests = len(asset_data)
    failed_tests = len(asset_data[asset_data['TestResult'] != 'P'])
    success_tests = total_tests - failed_tests

    failure_rate = (failed_tests / total_tests) * 100 if total_tests > 0 else 0
    success_rate = (success_tests / total_tests) * 100 if total_tests > 0 else 0

    asset_data_sorted = asset_data.sort_values('WorkCompletedOn')
    dates = asset_data_sorted['WorkCompletedOn'].tolist()
    intervals = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]

    optimal_days = calculate_optimal_interval(intervals, failure_rate)

    return {
        'asset_name': asset_name,
        'total_wells': len(wells),
        'total_tests': total_tests,
        'failed_tests': failed_tests,
        'success_tests': success_tests,
        'failure_rate': round(failure_rate, 2),
        'success_rate': round(success_rate, 2),
        'optimal_days': round(optimal_days, 0),
        'wells': wells
    }

def get_well_details(well_name, asset_name):
    well_data = df_data[(df_data['WellName'] == well_name) &
                        (df_data['FieldInstallationName'] == asset_name)].copy()
    if well_data.empty:
        return None

    well_data = well_data.sort_values('WorkCompletedOn')
    total_tests = len(well_data)
    failed_tests = len(well_data[well_data['TestResult'] != 'P'])
    failure_rate = (failed_tests / total_tests) * 100 if total_tests > 0 else 0

    dates = well_data['WorkCompletedOn'].tolist()
    intervals = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]

    optimal_days = calculate_optimal_interval(intervals, failure_rate)
    last_test_date = dates[-1] if dates else datetime.now()
    next_due_date = last_test_date + timedelta(days=optimal_days)

    ml_optimal_days = optimal_days
    ml_failure_rate = failure_rate

    if ml_model_optimal and ml_model_failure and len(intervals) > 0:
        try:
            avg_interval = np.mean(intervals)
            max_interval = max(intervals)
            min_interval = min(intervals)
            std_interval = np.std(intervals)
            avg_start_pressure = well_data['StartPressure'].mean()
            avg_finish_pressure = well_data['FinishPressure'].mean()
            pressure_variance = well_data['StartPressure'].var()

            features = np.array([[total_tests, failure_rate, avg_interval, max_interval,
                                  min_interval, std_interval, avg_start_pressure,
                                  avg_finish_pressure, pressure_variance]])

            features_scaled_optimal = scaler_optimal.transform(features)
            features_scaled_failure = scaler_failure.transform(features)

            ml_optimal_days = ml_model_optimal.predict(features_scaled_optimal)[0]
            ml_failure_rate = ml_model_failure.predict(features_scaled_failure)[0]

        except Exception as e:
            print(f"ML prediction error: {e}")

    test_history = []
    for _, row in well_data.iterrows():
        test_history.append({
            'date': row['WorkCompletedOn'].strftime('%Y-%m-%d'),
            'result': row['TestResult'],
            'test_type': row['TestTypeCode'],
            'start_pressure': row['StartPressure'],
            'finish_pressure': row['FinishPressure']
        })

    return {
        'well_name': well_name,
        'asset_name': asset_name,
        'total_tests': total_tests,
        'failed_tests': failed_tests,
        'failure_rate': round(failure_rate, 2),
        'optimal_days': round(optimal_days, 0),
        'ml_optimal_days': round(ml_optimal_days, 0),
        'ml_failure_rate': round(ml_failure_rate, 2),
        'last_test_date': last_test_date.strftime('%Y-%m-%d'),
        'next_due_date': next_due_date.strftime('%Y-%m-%d'),
        'test_history': test_history,
        'intervals': intervals
    }

@app.route('/')
def index():
    return render_template('dashboard.html', assets=all_assets)

@app.route('/api/asset/<asset_name>')
def get_asset_data(asset_name):
    overview = get_asset_overview(asset_name)
    return jsonify(overview)

@app.route('/api/well/<asset_name>/<well_name>')
def get_well_data(asset_name, well_name):
    details = get_well_details(well_name, asset_name)
    return jsonify(details)

@app.route('/api/retrain')
def retrain_models():
    train_ml_models()
    return jsonify({'status': 'success', 'message': 'Models retrained successfully'})

# ✅ SAFER INITIALIZATION FOR RENDER
load_and_process_data()
train_ml_models()

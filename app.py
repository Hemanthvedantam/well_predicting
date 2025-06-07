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
    """Load and process the Excel data with missing value handling"""
    global df_data, all_assets, asset_well_map, well_asset_map

    try:
        # Try different possible file paths
        possible_paths = [
            'well_data.xlsx',
            os.path.join(os.path.dirname(__file__), 'well_data.xlsx'),
            os.path.join(os.getcwd(), 'well_data.xlsx')
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if not csv_path:
            # Create sample data if file doesn't exist
            print("Warning: well_data.xlsx not found. Creating sample data...")
            create_sample_data()
            return
            
        print(f"Loading data from: {csv_path}")
        df_data = pd.read_excel(csv_path, engine='openpyxl')
        
        # Clean column names
        df_data.columns = df_data.columns.str.strip()
        print(f"Columns found: {list(df_data.columns)}")
        
        # Handle missing values
        df_data = handle_missing_values(df_data)
        
        # Extract unique assets
        if 'FieldInstallationName' in df_data.columns:
            all_assets = sorted(df_data['FieldInstallationName'].dropna().unique().tolist())
        else:
            print("Error: 'FieldInstallationName' column not found")
            return
            
        # Build asset-well mappings
        asset_well_map = {}
        well_asset_map = {}
        
        for asset in all_assets:
            if 'WellName' in df_data.columns:
                wells = sorted(df_data[df_data['FieldInstallationName'] == asset]['WellName'].dropna().unique().tolist())
                asset_well_map[asset] = wells
                for well in wells:
                    well_asset_map[well] = asset
            else:
                print("Error: 'WellName' column not found")
                return
        
        print(f"Loaded {len(all_assets)} assets with {len(well_asset_map)} wells total")
        print(f"Sample assets: {all_assets[:3] if len(all_assets) > 3 else all_assets}")
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        create_sample_data()

def create_sample_data():
    """Create sample data for testing"""
    global df_data, all_assets, asset_well_map, well_asset_map
    
    print("Creating sample data...")
    
    # Create sample data
    sample_data = []
    assets = ['Asset_A', 'Asset_B', 'Asset_C']
    
    for asset in assets:
        wells = [f'{asset}_Well_{i}' for i in range(1, 6)]  # 5 wells per asset
        
        for well in wells:
            # Generate test history for each well
            base_date = datetime.now() - timedelta(days=365)
            
            for i in range(10):  # 10 tests per well
                test_date = base_date + timedelta(days=i*30 + np.random.randint(-5, 5))
                
                sample_data.append({
                    'FieldInstallationName': asset,
                    'WellName': well,
                    'WorkCompletedOn': test_date,
                    'TestResult': np.random.choice(['P', 'F'], p=[0.8, 0.2]),
                    'TestTypeCode': np.random.choice(['Standard', 'Emergency', 'Scheduled']),
                    'StartPressure': np.random.normal(100, 20),
                    'FinishPressure': np.random.normal(95, 15),
                    'Maintained': np.random.choice(['TRUE', 'FALSE'], p=[0.7, 0.3]),
                    'ManualOverride': np.random.choice(['TRUE', 'FALSE'], p=[0.1, 0.9])
                })
    
    df_data = pd.DataFrame(sample_data)
    df_data = handle_missing_values(df_data)
    
    all_assets = sorted(df_data['FieldInstallationName'].unique().tolist())
    asset_well_map = {}
    well_asset_map = {}
    
    for asset in all_assets:
        wells = sorted(df_data[df_data['FieldInstallationName'] == asset]['WellName'].unique().tolist())
        asset_well_map[asset] = wells
        for well in wells:
            well_asset_map[well] = asset
    
    print(f"Created sample data with {len(all_assets)} assets and {len(well_asset_map)} wells")

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    try:
        # Convert date column
        if 'WorkCompletedOn' in df.columns:
            df['WorkCompletedOn'] = pd.to_datetime(df['WorkCompletedOn'], errors='coerce')
            df['WorkCompletedOn'].fillna(pd.Timestamp.now(), inplace=True)
        
        # Handle categorical columns
        categorical_columns = ['Maintained', 'ManualOverride', 'TestResult', 'TestTypeCode']
        for col in categorical_columns:
            if col in df.columns:
                if col == 'Maintained' or col == 'ManualOverride':
                    df[col].fillna('FALSE', inplace=True)
                elif col == 'TestResult':
                    df[col].fillna('P', inplace=True)
                elif col == 'TestTypeCode':
                    df[col].fillna('Standard', inplace=True)
        
        # Handle numerical columns
        numerical_columns = ['StartPressure', 'FinishPressure']
        for col in numerical_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    except Exception as e:
        print(f"Error handling missing values: {str(e)}")
        return df

def calculate_optimal_interval(intervals, failure_rate):
    """Calculate optimal test interval based on historical data"""
    if not intervals or len(intervals) == 0:
        return 90
    
    avg_interval = np.mean(intervals)
    safety_factor = 0.9
    
    # Adjust safety factor based on failure rate
    if failure_rate > 20:
        safety_factor = 0.6
    elif failure_rate > 10:
        safety_factor = 0.75
    
    optimal = avg_interval * safety_factor
    return max(30, min(365, optimal))

def train_ml_models():
    """Train machine learning models for predictions"""
    global ml_model_optimal, ml_model_failure, scaler_optimal, scaler_failure

    if df_data is None or len(df_data) == 0:
        print("No data available for training ML models")
        return

    try:
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
                    
                    # Safely get pressure data
                    start_pressures = well_data['StartPressure'].dropna()
                    finish_pressures = well_data['FinishPressure'].dropna()
                    
                    avg_start_pressure = start_pressures.mean() if len(start_pressures) > 0 else 100
                    avg_finish_pressure = finish_pressures.mean() if len(finish_pressures) > 0 else 95
                    pressure_variance = start_pressures.var() if len(start_pressures) > 0 else 10

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
            
            print("ML models trained successfully")
        else:
            print("Insufficient data for ML model training")
            
    except Exception as e:
        print(f"Error training ML models: {str(e)}")

def get_asset_overview(asset_name):
    """Get overview statistics for an asset"""
    try:
        if df_data is None or asset_name not in all_assets:
            return None

        asset_data = df_data[df_data['FieldInstallationName'] == asset_name]
        if asset_data.empty:
            return None

        wells = asset_well_map.get(asset_name, [])
        total_tests = len(asset_data)
        failed_tests = len(asset_data[asset_data['TestResult'] != 'P'])
        success_tests = total_tests - failed_tests

        failure_rate = (failed_tests / total_tests) * 100 if total_tests > 0 else 0
        success_rate = (success_tests / total_tests) * 100 if total_tests > 0 else 0

        # Calculate optimal interval for the asset
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
    except Exception as e:
        print(f"Error getting asset overview: {str(e)}")
        return None

def get_well_details(well_name, asset_name):
    """Get detailed information for a specific well"""
    try:
        if df_data is None:
            return None
            
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

        # Initialize ML predictions with default values
        ml_optimal_days = optimal_days
        ml_failure_rate = failure_rate

        # Try ML predictions if models are available
        if ml_model_optimal and ml_model_failure and len(intervals) > 0:
            try:
                avg_interval = np.mean(intervals)
                max_interval = max(intervals)
                min_interval = min(intervals)
                std_interval = np.std(intervals)
                
                # Safely get pressure data
                start_pressures = well_data['StartPressure'].dropna()
                finish_pressures = well_data['FinishPressure'].dropna()
                
                avg_start_pressure = start_pressures.mean() if len(start_pressures) > 0 else 100
                avg_finish_pressure = finish_pressures.mean() if len(finish_pressures) > 0 else 95
                pressure_variance = start_pressures.var() if len(start_pressures) > 0 else 10

                features = np.array([[total_tests, failure_rate, avg_interval, max_interval,
                                      min_interval, std_interval, avg_start_pressure,
                                      avg_finish_pressure, pressure_variance]])

                features_scaled_optimal = scaler_optimal.transform(features)
                features_scaled_failure = scaler_failure.transform(features)

                ml_optimal_days = ml_model_optimal.predict(features_scaled_optimal)[0]
                ml_failure_rate = ml_model_failure.predict(features_scaled_failure)[0]

            except Exception as e:
                print(f"ML prediction error: {e}")

        # Prepare test history
        test_history = []
        for _, row in well_data.iterrows():
            test_history.append({
                'date': row['WorkCompletedOn'].strftime('%Y-%m-%d'),
                'result': row['TestResult'],
                'test_type': row.get('TestTypeCode', 'Standard'),
                'start_pressure': float(row['StartPressure']) if pd.notna(row['StartPressure']) else 0,
                'finish_pressure': float(row['FinishPressure']) if pd.notna(row['FinishPressure']) else 0
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
    except Exception as e:
        print(f"Error getting well details: {str(e)}")
        return None

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('dashboard.html', assets=all_assets)

@app.route('/api/assets')
def get_assets():
    """Get list of all assets"""
    return jsonify(all_assets)

@app.route('/api/asset/<asset_name>')
def get_asset_data(asset_name):
    """Get data for a specific asset"""
    overview = get_asset_overview(asset_name)
    if overview:
        return jsonify(overview)
    else:
        return jsonify({'error': 'Asset not found'}), 404

@app.route('/api/wells/<asset_name>')
def get_wells_for_asset(asset_name):
    """Get list of wells for a specific asset"""
    wells = asset_well_map.get(asset_name, [])
    return jsonify(wells)

@app.route('/api/well/<asset_name>/<well_name>')
def get_well_data(asset_name, well_name):
    """Get data for a specific well"""
    details = get_well_details(well_name, asset_name)
    if details:
        return jsonify(details)
    else:
        return jsonify({'error': 'Well not found'}), 404

@app.route('/api/retrain')
def retrain_models():
    """Retrain ML models"""
    try:
        train_ml_models()
        return jsonify({'status': 'success', 'message': 'Models retrained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/debug')
def debug_info():
    """Debug endpoint to check data status"""
    return jsonify({
        'data_loaded': df_data is not None,
        'total_records': len(df_data) if df_data is not None else 0,
        'assets_count': len(all_assets),
        'wells_count': len(well_asset_map),
        'columns': list(df_data.columns) if df_data is not None else [],
        'sample_assets': all_assets[:5] if all_assets else []
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize the application
if __name__ == '__main__':
    print("Starting Well Analysis Dashboard...")
    try:
        load_and_process_data()
        train_ml_models()
        print(f"Dashboard initialized with {len(all_assets)} assets")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        # Still start the app with sample data
        create_sample_data()
        train_ml_models()
        app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # For production deployment (like Render)
    load_and_process_data()
    train_ml_models()

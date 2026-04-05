import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --- 1. App Header & Marketing Pitch ---
st.set_page_config(page_title="Ski Tracker - Low Power IMU", layout="wide")
st.title("⛷️ Low-Power Ski Surface Classifier")
st.markdown("""
Unlike traditional GPS trackers that drain your battery on the mountain, this app uses your phone's low-power IMU sensors (Accelerometer, Gyroscope, and Barometer) to track and classify your ski runs. 
Upload your sensor data below to see a breakdown of your day!
""")

# --- 2. Load the Model ---
@st.cache_resource
def load_model():
    with open('ski_knn_model.pkl', 'rb') as f:
        return pickle.load(f)

knn_model = load_model()

# --- 3. File Uploaders ---
col1, col2, col3 = st.columns(3)
with col1:
    accel_file = st.file_uploader("Upload Accelerometer.csv", type=['csv'])
with col2:
    gyro_file = st.file_uploader("Upload Gyroscope.csv", type=['csv'])
with col3:
    baro_file = st.file_uploader("Upload Barometer.csv", type=['csv'])

if accel_file and gyro_file and baro_file:
    st.success("All files uploaded successfully! Processing data...")
    
    # Load data
    df_accel = pd.read_csv(accel_file)
    df_gyro = pd.read_csv(gyro_file)
    df_baro = pd.read_csv(baro_file)
    
    # Merge Data
    df_merged = pd.merge(
        df_accel[['seconds_elapsed', 'x', 'y', 'z']],
        df_gyro[['seconds_elapsed', 'x', 'y', 'z']],
        on='seconds_elapsed', how='outer', suffixes=('_accel', '_gyro')
    )
    df_merged = pd.merge(df_merged, df_baro[['seconds_elapsed', 'pressure']], on='seconds_elapsed', how='outer')
    
    # Interpolate and calculate magnitudes
    df_merged['pressure'] = df_merged['pressure'].interpolate(method='linear', limit_direction='both')
    df_merged['accelerometer_magnitude'] = np.sqrt(df_merged['x_accel']**2 + df_merged['y_accel']**2 + df_merged['z_accel']**2)
    df_merged['gyroscope_magnitude'] = np.sqrt(df_merged['x_gyro']**2 + df_merged['y_gyro']**2 + df_merged['z_gyro']**2)
    df_merged = df_merged.sort_values(by='seconds_elapsed').reset_index(drop=True)
    
    # Windowing Logic
    window_size_seconds = 10
    step_size_seconds = 1
    current_time = df_merged['seconds_elapsed'].min()
    max_time = df_merged['seconds_elapsed'].max()
    
    test_features_list = []
    times_list = []
    
    while current_time < max_time:
        window_start = current_time
        window_end = current_time + window_size_seconds
        window_data = df_merged[(df_merged['seconds_elapsed'] >= window_start) & (df_merged['seconds_elapsed'] < window_end)]
        
        if len(window_data) > 10 and not window_data['pressure'].isnull().all():
            test_features_list.append({
                'mean_accelerometer_magnitude': window_data['accelerometer_magnitude'].mean(),
                'mean_gyroscope_magnitude': window_data['gyroscope_magnitude'].mean(),
                'pressure_change': window_data['pressure'].iloc[-1] - window_data['pressure'].iloc[0]
            })
            times_list.append({'start_time': window_start, 'end_time': window_end})
            
        current_time += step_size_seconds
        
    df_test_features = pd.DataFrame(test_features_list)
    
    # --- 4. Prediction ---
    predictions = knn_model.predict(df_test_features)
    df_results = pd.DataFrame(times_list)
    df_results['classification'] = predictions
    
    st.subheader("Classification Results")
    
    # Summary Metrics
    lift_time = len(df_results[df_results['classification'] == 'lift']) * step_size_seconds
    groomed_time = len(df_results[df_results['classification'] == 'groomed']) * step_size_seconds
    powder_time = len(df_results[df_results['classification'] == 'powder']) * step_size_seconds
    
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric("Lift Time", f"{lift_time} sec")
    scol2.metric("Groomed Run Time", f"{groomed_time} sec")
    scol3.metric("Powder Run Time", f"{powder_time} sec")

    # --- 5. Visualization ---
    st.subheader("Run Timeline")
    fig, ax = plt.subplots(figsize=(10, 2))
    color_map = {'lift': 'skyblue', 'groomed': 'lightgreen', 'powder': 'lightcoral'}
    
    for index, row in df_results.iterrows():
        ax.axvspan(row['start_time'], row['end_time'], color=color_map.get(row['classification'], 'gray'), alpha=0.6)
        
    handles = [plt.Rectangle((0,0),1,1, color=color_map[c]) for c in color_map]
    ax.legend(handles, color_map.keys(), loc='upper right')
    ax.set_xlabel("Time (seconds)")
    ax.set_yticks([])
    st.pyplot(fig)
else:
    st.info("Please upload all three CSV files to see your run analysis.")

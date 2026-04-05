import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import zipfile

# --- 1. App Header & Instructions ---
st.set_page_config(page_title="Ski Tracker - Low Power IMU", layout="wide")
st.title("⛷️ Low-Power Ski Surface Classifier")

# The instructions are written using st.markdown to allow for bolding and bullet points
st.markdown("""
Unlike traditional GPS trackers that drain your battery on the mountain, this app uses your phone's low-power IMU sensors to track and classify your ski runs as on the lift, on a powder run, and on a groomed run. 

### How to use this app:
1. Download the **Sensor Logger** app on your iOS or Android device.
2. Open the app and toggle **ON** the following three sensors: **Accelerometer**, **Gyroscope**, and **Barometer**.
3. Hit record, securely stash your phone in a pocket, and go ski!
4. When you are done, stop the recording and export the data as a `.zip` file.
5. Upload that `.zip` file right here for instant analysis.
""")

# --- 2. Load the Model ---
@st.cache_resource
def load_model():
    with open('ski_knn_model.pkl', 'rb') as f:
        return pickle.load(f)

knn_model = load_model()

# --- 3. File Uploader (ZIP Handling) ---
st.subheader("Upload Your Run Data")
uploaded_zip = st.file_uploader("Upload your Sensor Logger .zip file", type=['zip'])

if uploaded_zip is not None:
    with st.spinner("Unzipping and processing data..."):
        try:
            # Open the zip file in memory
            with zipfile.ZipFile(uploaded_zip) as z:
                # Search through the zip to find the exact files we need
                accel_filename = next((f for f in z.namelist() if f.endswith('Accelerometer.csv')), None)
                gyro_filename = next((f for f in z.namelist() if f.endswith('Gyroscope.csv')), None)
                baro_filename = next((f for f in z.namelist() if f.endswith('Barometer.csv')), None)
                
                if not (accel_filename and gyro_filename and baro_filename):
                    st.error("Error: The uploaded zip file does not contain all three required files (Accelerometer.csv, Gyroscope.csv, Barometer.csv).")
                    st.stop()
                
                # Read the CSVs directly from the zip file into Pandas
                with z.open(accel_filename) as f:
                    df_accel = pd.read_csv(f)
                with z.open(gyro_filename) as f:
                    df_gyro = pd.read_csv(f)
                with z.open(baro_filename) as f:
                    df_baro = pd.read_csv(f)
                    
            st.success("Data successfully extracted! Analyzing your runs...")
            
            # --- 4. Data Merging & Processing ---
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
            
            # --- 5. Prediction & Display ---
            predictions = knn_model.predict(df_test_features)
            df_results = pd.DataFrame(times_list)
            df_results['classification'] = predictions
            
            st.markdown("---")
            st.subheader("Classification Results")
            
            # Summary Metrics
            lift_time = len(df_results[df_results['classification'] == 'lift']) * step_size_seconds
            groomed_time = len(df_results[df_results['classification'] == 'groomed']) * step_size_seconds
            powder_time = len(df_results[df_results['classification'] == 'powder']) * step_size_seconds
            
            scol1, scol2, scol3 = st.columns(3)
            scol1.metric("Lift Time", f"{lift_time} sec")
            scol2.metric("Groomed Run Time", f"{groomed_time} sec")
            scol3.metric("Powder Run Time", f"{powder_time} sec")

            # --- 6. Visualization ---
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
            
        except Exception as e:
            st.error(f"An error occurred while processing the zip file: {e}")

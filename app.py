import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from meteostat import Point, Daily
from sklearn.linear_model import LinearRegression
import warnings
import io
import pmdarima as pm
from matplotlib.dates import date2num
from statistics import mode
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import pickle
import os

# Disable a specific pandas warning
warnings.filterwarnings('ignore')
register_matplotlib_converters()

# --- Hard-coded Configuration ---
API_KEY = "97564c78d6a7b0723ad21e6c6d3b8dee"
LATITUDE = 4.7156
LONGITUDE = 8.125
RISK_MODEL_PATH = "flood_risk_model.pkl"
PREDICTION_MODEL_PATH = "flood_prediction_model.pkl"
TIDE_ARIMA_MODEL_PATH = "tide_arima_model.pkl"
# --- End Configuration ---

# Page configuration
st.set_page_config(
    page_title="Flood Risk Monitor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Functions
@st.cache_data(ttl=3600)
def get_current_weather(lat, lon, api_key):
    """Fetch current weather data from OpenWeatherMap"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching current weather: {str(e)}")
        return None

@st.cache_data(ttl=21600)
def get_historical_weather(lat, lon, days=30):
    """Fetch historical weather data using Meteostat and interpolate missing values."""
    try:
        location = Point(lat, lon)
        end = datetime.now()
        start = end - timedelta(days=days)
        data = Daily(location, start, end)
        data = data.fetch()

        if data.empty:
            return None

        # --- FIX: Interpolate missing values in 'prcp' and 'tavg' columns ---
        data['prcp'] = data['prcp'].interpolate(method='linear', limit_direction='both')
        data['tavg'] = data['tavg'].interpolate(method='linear', limit_direction='both')
        # --- End FIX ---

        return data
    except Exception as e:
        st.error(f"Error fetching historical weather: {str(e)}")
        return None

def train_and_save_risk_model(historical_data):
    """Train a simple linear regression model for flood risk and save it."""
    df = historical_data.copy()
    df.dropna(subset=['prcp'], inplace=True)
    if df.empty:
        st.error("No valid historical data available for risk model training.")
        return None
    df['historical_risk_score'] = 0
    df.loc[df['prcp'] > 50, 'historical_risk_score'] = 40
    df.loc[(df['prcp'] > 20) & (df['prcp'] <= 50), 'historical_risk_score'] = 20
    df.loc[(df['prcp'] > 5) & (df['prcp'] <= 20), 'historical_risk_score'] = 10
    X = df[['prcp']]
    y = df['historical_risk_score']
    model = LinearRegression()
    model.fit(X, y)
    with open(RISK_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model

def predict_flood_risk(model, current_precip):
    """Predict flood risk using the trained ML model."""
    if model is None:
        return 0, "UNKNOWN", []
    predicted_score = model.predict([[current_precip]])[0]
    predicted_score = max(0, min(100, predicted_score))
    risk_factors = []
    if current_precip > 50:
        risk_factors.append("Heavy current precipitation")
    elif current_precip > 20:
        risk_factors.append("Moderate current precipitation")
    elif current_precip > 5:
        risk_factors.append("Light current precipitation")
    if predicted_score >= 70:
        risk_level = "HIGH"
    elif predicted_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    return predicted_score, risk_level, risk_factors

def create_precipitation_trend_chart(historical_data):
    """Create precipitation trend chart"""
    if historical_data is None or historical_data.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['prcp'],
        mode='lines+markers',
        name='Daily Precipitation',
        line=dict(color='#2c3e50', width=2),
        marker=dict(size=6, color='#3498db')
    ))
    fig.update_layout(
        title="Precipitation Trend Analysis (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title="Precipitation (mm)",
        hovermode='x unified',
        height=400
    )
    return fig

def create_temperature_trend_chart(historical_data):
    """Create temperature trend chart"""
    if historical_data is None or historical_data.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['tavg'],
        mode='lines+markers',
        name='Daily Average Temperature',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=6, color='#e74c3c')
    ))
    fig.update_layout(
        title="Temperature Trend Analysis (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title="Average Temperature (°C)",
        hovermode='x unified',
        height=400
    )
    return fig

def create_risk_gauge(risk_score):
    """Create risk gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Flood Risk Score"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#34495e"},
            'steps': [
                {'range': [0, 40], 'color': "#2ecc71"},
                {'range': [40, 70], 'color': "#f39c12"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# --- Tide Prediction Functions ---
def inputDict1(raw_data, time_col, tidal_col):
    data = raw_data.copy()
    time_column_name = time_col
    try:
        data[time_column_name] = pd.to_datetime(data[time_column_name], dayfirst=True)
    except ValueError:
        st.error("Could not convert the time column to datetime. Please ensure the column contains valid dates and times.")
        st.stop()
    data.index = data[time_column_name]
    data = data.sort_index()
    data[tidal_col] = data[tidal_col].interpolate(method='linear')
    data[tidal_col] = data[tidal_col].bfill()
    data[tidal_col] = data[tidal_col].ffill()
    time_array = data.index
    start_time = time_array[:-1]
    end_time = time_array[1:]
    time_diff_array = end_time - start_time
    time_diff = mode(time_diff_array)
    time_diff_float = time_diff.total_seconds() / 60
    time_gap_div = np.where((time_diff_array > time_diff) & ((time_diff_array % time_diff) == pd.Timedelta(0)))
    time_gap_undiv = np.where((time_diff_array > time_diff) & ((time_diff_array % time_diff) != pd.Timedelta(0)))
    start_gap_div = start_time[time_gap_div] + time_diff
    end_gap_div = end_time[time_gap_div] - time_diff
    start_gap_undiv = start_time[time_gap_undiv] + time_diff
    end_gap_undiv = end_time[time_gap_undiv]
    data_dummy = []
    for i in range(len(start_gap_div)):
        time_add = pd.date_range(start=start_gap_div[i], end=end_gap_div[i], freq=f'{int(time_diff.total_seconds() // 60)}min')
        nan_add = pd.DataFrame({time_column_name: time_add, tidal_col: pd.Series(np.nan, index=list(range(len(time_add))))})
        nan_add.index = nan_add[time_column_name]
        nan_add = nan_add.iloc[:, 1:]
        data_dummy.append(nan_add)
    for i in range(len(start_gap_undiv)):
        time_add = pd.date_range(start=start_gap_undiv[i], end=end_gap_undiv[i], freq=f'{int(time_diff.total_seconds() // 60)}min')
        nan_add = pd.DataFrame({time_column_name: time_add, tidal_col: pd.Series(np.nan, index=list(range(len(time_add))))})
        nan_add.index = nan_add[time_column_name]
        nan_add = nan_add.iloc[:, 1:]
        data_dummy.append(nan_add)
    if len(data_dummy) > 0:
        data_add = pd.concat(data_dummy, sort=True)
        filled = pd.concat([data, data_add], sort=True)
    else:
        filled = data.copy()
    filled = filled.sort_index()
    filled[tidal_col] = filled[tidal_col].interpolate(method='linear')
    time_array2 = filled.index
    depth_array2 = filled[tidal_col].values
    input_dict = {'depth': depth_array2, 'time': time_array2, 'interval': time_diff_float}
    return input_dict

try:
    from ttide import t_tide
except ImportError:
    st.error("The 'ttide' library is not installed. Please install it with 'pip install ttide' to use the tide prediction feature.")
    t_tide = None

def ttideAnalyse(raw_data, time_col, tidal_col, latitude):
    if t_tide is None:
        return None, "ttide not installed"
    input_dict = inputDict1(raw_data, time_col, tidal_col)
    ad = input_dict['depth']
    at = input_dict['time']
    time_diff = input_dict['interval'] / 60
    time_num = date2num(at.to_pydatetime())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with io.StringIO() as buf, redirect_stdout(buf):
            coef = t_tide(ad, dt=time_diff, stime=time_num[0], lat=latitude, synth=0)
            ttide_report = buf.getvalue()
    return coef, ttide_report

def ttidePredict(raw_data, start_date, end_date, interval, time_col, tidal_col, latitude):
    raw_data[time_col] = pd.to_datetime(raw_data[time_col], dayfirst=True)
    predic_time = pd.date_range(start=start_date, end=end_date, freq=interval)

    input_dict = inputDict1(raw_data, time_col, tidal_col)
    coef, _ = ttideAnalyse(raw_data, time_col, tidal_col, latitude)
    if coef is None:
        st.stop()
    msl = coef['z0']

    observed_time_num = date2num(input_dict['time'].to_pydatetime())
    astro_pred_observed = coef(observed_time_num) + msl
    residuals = input_dict['depth'] - astro_pred_observed

    n_periods = len(predic_time)
    residual_pred_future = np.zeros(n_periods)

    if os.path.exists(TIDE_ARIMA_MODEL_PATH):
        st.write("Loading pre-trained ARIMA model for tide residuals...")
        with open(TIDE_ARIMA_MODEL_PATH, 'rb') as f:
            arima_model = pickle.load(f)
    else:
        st.write("Fitting Auto ARIMA model to residuals...")
        try:
            residuals_series = pd.Series(residuals)
            arima_model = pm.auto_arima(residuals_series.dropna(), seasonal=False, stepwise=True,
                                        suppress_warnings=True, error_action='ignore')
            with open(TIDE_ARIMA_MODEL_PATH, 'wb') as f:
                pickle.dump(arima_model, f)
        except Exception as e:
            st.error(f"Auto ARIMA model failed: {e}. Cannot proceed with hybrid model.")
            st.stop()

    residual_pred_future = arima_model.predict(n_periods=n_periods)

    astro_pred_future = coef(date2num(predic_time.to_pydatetime())) + msl
    if astro_pred_future.shape[0] != residual_pred_future.shape[0]:
        st.error("Prediction failed: Harmonic prediction and residual prediction have different lengths.")
        st.stop()

    predic = astro_pred_future + residual_pred_future
    prediction_df = pd.DataFrame({'Time': predic_time, 'Predicted Tide': predic})

    st.write(f"Tide Predictions using Hybrid Model:")
    st.write(prediction_df)
    st.line_chart(prediction_df.set_index('Time')['Predicted Tide'])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predic_time, predic, label='Hybrid Predicted Tide', color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel('Tide Level')
    ax.set_title(f'Hybrid Tide Predictions from {start_date} to {end_date}')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    return prediction_df
# --- End Tide Prediction Functions ---

# --- Sidebar Navigation ---
st.sidebar.title("Dashboard Navigation")
selected_page = st.sidebar.radio("Go to", ["Flood Risk Monitoring", "Future Flood Prediction", "Tide Prediction Analysis"])

# --- Main Page Content ---
if selected_page == "Flood Risk Monitoring":
    st.title("🌊 Flood Risk Monitoring System")
    st.markdown("Real-time flood risk assessment and weather monitoring.")
    st.sidebar.header("🌍 Location")
    st.sidebar.info(f"Monitoring location: ({LATITUDE:.2f}, {LONGITUDE:.2f})")

    current_weather = get_current_weather(LATITUDE, LONGITUDE, API_KEY)
    historical_data = get_historical_weather(LATITUDE, LONGITUDE)

    if current_weather and historical_data is not None and not historical_data.empty:
        temp = current_weather['main']['temp']
        wind_speed = current_weather['wind']['speed']
        precipitation = current_weather.get('rain', {}).get('1h', 0)

        # Load or train the flood risk model
        if os.path.exists(RISK_MODEL_PATH):
            st.write("Loading pre-trained flood risk model...")
            with open(RISK_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        else:
            st.write("Training flood risk model...")
            model = train_and_save_risk_model(historical_data)

        if model:
            risk_score, risk_level, risk_factors = predict_flood_risk(model, precipitation)

            st.header("🌤️ Current Weather Conditions")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Temperature", f"{temp}°C")
            with col2:
                st.metric("Wind Speed", f"{wind_speed} m/s")
            with col3:
                st.metric("Precipitation", f"{precipitation} mm/h")

            st.header("🚨 Flood Risk Assessment")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Risk Level", risk_level)
                st.metric("Risk Score", f"{risk_score:.0f}/100")
                gauge_fig = create_risk_gauge(risk_score)
                st.plotly_chart(gauge_fig, use_container_width=True)
            with col2:
                st.subheader("Risk Factors")
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.write("• No significant risk factors detected")
                st.subheader("Recommendations")
                if risk_level == "HIGH":
                    st.error("⚠️ High flood risk detected! Avoid low-lying areas and monitor local alerts.")
                elif risk_level == "MEDIUM":
                    st.warning("⚡ Moderate flood risk. Stay alert and avoid flood-prone areas.")
                else:
                    st.success("✅ Low flood risk. Normal precautions apply.")

            st.header("📈 Trend Analysis")

            precipitation_chart = create_precipitation_trend_chart(historical_data)
            if precipitation_chart:
                st.plotly_chart(precipitation_chart, use_container_width=True)

            temperature_chart = create_temperature_trend_chart(historical_data)
            if temperature_chart:
                st.plotly_chart(temperature_chart, use_container_width=True)

            if 'prcp' in historical_data.columns and not historical_data['prcp'].isnull().all():
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("30-Day Statistics")
                    avg_precip = historical_data['prcp'].mean()
                    max_precip = historical_data['prcp'].max()
                    total_precip = historical_data['prcp'].sum()
                    st.metric("Average Daily Precipitation", f"{avg_precip:.1f} mm")
                    st.metric("Maximum Daily Precipitation", f"{max_precip:.1f} mm")
                    st.metric("Total Precipitation", f"{total_precip:.1f} mm")
                with col2:
                    st.subheader("Recent Trends")
                    recent_avg = historical_data['prcp'].tail(7).mean()
                    previous_avg = historical_data['prcp'].iloc[-14:-7].mean()
                    trend_change = recent_avg - previous_avg
                    st.metric("7-Day Average", f"{recent_avg:.1f} mm")
                    st.metric("Previous Week Average", f"{previous_avg:.1f} mm")
                    st.metric("Weekly Change", f"{trend_change:+.1f} mm")
                st.subheader("Precipitation Distribution")
                hist_fig = px.histogram(
                    historical_data,
                    x='prcp',
                    nbins=20,
                    title="Distribution of Daily Precipitation",
                    labels={'prcp': 'Precipitation (mm)', 'count': 'Frequency'}
                )
                st.plotly_chart(hist_fig, use_container_width=True)

            st.sidebar.markdown("---")
            if st.sidebar.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
            st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.error("Failed to fetch weather data. Please ensure the API key is valid and historical data is available.")

    st.markdown("""
    ---
  
    ### Risk Assessment Factors:
    -   Current precipitation intensity
    -   Wind speed conditions
    -   Recent rainfall patterns
    -   Soil saturation indicators
    """)

elif selected_page == "Future Flood Prediction":
    st.title("🔮 Future Flood Prediction")
    st.markdown("Predicting future flood risk based on forecasted precipitation using a machine learning model.")

    historical_data = get_historical_weather(LATITUDE, LONGITUDE, days=365) # Using more data for better prediction

    if historical_data is not None and not historical_data.empty:

        # Load or train the flood prediction model
        if os.path.exists(PREDICTION_MODEL_PATH):
            st.write("Loading pre-trained precipitation prediction model...")
            with open(PREDICTION_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        else:
            st.write("Training precipitation prediction model...")
            df = historical_data.dropna(subset=['prcp'])
            if df.empty:
                st.error("No valid historical precipitation data for model training.")
                st.stop()
            try:
                model = pm.auto_arima(df['prcp'], seasonal=False, stepwise=True,
                                    suppress_warnings=True, error_action='ignore')
                with open(PREDICTION_MODEL_PATH, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                st.error(f"Failed to train prediction model: {e}")
                st.stop()

        days_to_predict = st.slider("Select number of days to predict:", 1, 30, 7)

        forecast, conf_int = model.predict(n_periods=days_to_predict, return_conf_int=True)
        forecast_dates = pd.date_range(start=historical_data.index[-1] + timedelta(days=1), periods=days_to_predict)

        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Precipitation': forecast})

        st.subheader("Future Precipitation Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['prcp'], name='Observed', line=dict(color='#2c3e50')))
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Precipitation'], name='Forecast', line=dict(color='#3498db', dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=conf_int[:, 1], name='Upper Confidence Interval', fill='tonexty', fillcolor='rgba(52, 152, 219, 0.2)', line=dict(width=0)))
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=conf_int[:, 0], name='Lower Confidence Interval', fill='tonexty', fillcolor='rgba(52, 152, 219, 0.2)', line=dict(width=0)))

        fig.update_layout(title='Precipitation Forecast', xaxis_title='Date', yaxis_title='Precipitation (mm)', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        max_precip_forecast = forecast.max()
        avg_precip_forecast = forecast.mean()

        st.subheader("Predicted Flood Risk")

        if max_precip_forecast > 50:
            st.error("🚨 **High Risk:** The model predicts a high chance of heavy rainfall events within the forecast period.")
        elif max_precip_forecast > 20:
            st.warning("⚡ **Medium Risk:** Moderate precipitation is forecasted, which could lead to flooding in low-lying areas.")
        else:
            st.success("✅ **Low Risk:** The forecast indicates low precipitation, with a minimal risk of flooding.")

        st.info(f"The highest forecasted daily precipitation is {max_precip_forecast:.1f} mm, with an average of {avg_precip_forecast:.1f} mm per day over the next {days_to_predict} days.")

    else:
        st.error("Failed to fetch sufficient historical weather data to train the prediction model.")

elif selected_page == "Tide Prediction Analysis":
    st.title("🌊 Tide Prediction Analysis")


    try:
        raw_data = pd.read_csv('TideCompiled.csv')
        st.write("Raw Data Preview:")
        st.write(raw_data.head())

        with st.expander("Tide Prediction Tools", expanded=True):
            tab1, tab2 = st.tabs(["Tide Data", "Prediction Parameters"])
            with tab1:
                st.write("Select Columns for Time and Tidal Data")
                time_column = st.selectbox('Select Time Column', raw_data.columns, index=list(raw_data.columns).index('DateTime') if 'DateTime' in raw_data.columns else 0)
                tidal_column = st.selectbox('Select Tidal Value Column', raw_data.columns, index=list(raw_data.columns).index('Tide(m)') if 'Tide(m)' in raw_data.columns else 0)

            raw_data[time_column] = pd.to_datetime(raw_data[time_column], dayfirst=True)

            with tab2:
                st.write("Input Prediction Parameters")
                latitude = st.number_input("Enter Latitude:", min_value=-90.0, max_value=90.0, value=4.7156)
                st.info("The model is trained on the full historical dataset to generate the most accurate predictions. You can select any date range below to visualize the results.")
                start_date = st.date_input("Start date for prediction:", value=raw_data[time_column].min().date())
                end_date = st.date_input("End date for prediction:", value=raw_data[time_column].max().date() + timedelta(days=30))
                interval_options = {"1 Hour": 'h', "30 Minutes": '30min', "15 Minutes": '15min'}
                interval = st.selectbox("Select Prediction Interval", list(interval_options.keys()))

            if st.button("Predict Tides"):
                if start_date >= end_date:
                    st.error("The start date must be before the end date. Please adjust your selection.")
                else:
                    prediction_df = ttidePredict(raw_data, start_date, end_date, interval_options[interval], time_column, tidal_column, latitude)

                    coef, ttide_report = ttideAnalyse(raw_data, time_column, tidal_column, latitude)
                    if ttide_report:
                        buffer = io.BytesIO()
                        buffer.write(ttide_report.encode())
                        buffer.seek(0)
                        st.download_button(
                            label="Download TTide Report as .txt",
                            data=buffer,
                            file_name="TTide_Report.txt",
                            mime="text/plain"
                        )

            if st.checkbox('Plot Observed Data'):
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(raw_data[time_column], raw_data[tidal_column], label='Observed Tide')
                ax.set_xlabel('Time')
                ax.set_ylabel('Tide Level')
                ax.set_title('Observed Tide Data')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
    except FileNotFoundError:
        st.error(f"Error: The file could not be found. Please ensure the file 'TideCompiled.csv' exists at the specified path on your local machine.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.info("Please ensure your CSV file has the correct columns and a valid format.")

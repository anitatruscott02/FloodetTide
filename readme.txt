Flood Risk and Tide Prediction Dashboard
This is a Streamlit application designed to monitor flood risk and analyze tide predictions. It combines real-time weather data with historical tide information to provide a comprehensive dashboard.

Features
Flood Risk Monitoring: Fetches current weather data (temperature, wind speed, precipitation) to calculate and display a real-time flood risk score.

Precipitation and Temperature Trend Analysis: Visualizes historical weather trends over the last 30 days.

Tide Prediction Analysis: Uses a hybrid model (harmonic analysis with ARIMA for residuals) to predict future tide levels based on a historical dataset.

File Structure
app.py: The main Streamlit application script.

requirements.txt: Lists all the necessary Python libraries to run the application.

TideCompiled.csv: A sample dataset for the tide prediction module.

Installation
Clone the repository:

git clone <your-repo-url>
cd <your-repo-folder>

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required packages:

pip install -r requirements.txt

Usage
Obtain an API Key: The flood risk module requires a free API key from OpenWeatherMap. Replace the placeholder in app.py with your key:

API_KEY = "your_api_key_here"

Run the application:

streamlit run app.py

The application will open in your web browser, allowing you to navigate between the Flood Risk Monitoring and Tide Prediction Analysis pages.

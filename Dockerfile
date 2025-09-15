# Start with your base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary data and model files into the container
COPY TideCompiled.csv .
COPY flood_risk_model.pkl .
COPY flood_prediction_model.pkl .
COPY tide_arima_model.pkl .
COPY advanced_flood_classifier.pkl .
COPY advanced_tide_regressor.pkl .

# Copy the rest of your application code
COPY . .

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]

# Start with your base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the zipped ttide library and install the utility to extract it
COPY ttide_py-master.rar .
RUN apt-get update && apt-get install -y --no-install-recommends unrar && rm -rf /var/lib/apt/lists/*
RUN unrar x ttide_py-master.rar && rm ttide_py-master.rar

# Run the installation command from inside the extracted folder
RUN python setup.py install

# Copy all necessary data and model files into the container
COPY TideCompiled.csv .
COPY flood_risk_model.pkl .
COPY flood_prediction_model.pkl .
COPY tide_arima_model.pkl .

# Copy the rest of your application code
COPY . .

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]

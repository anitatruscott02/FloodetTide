# Start with the base Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install git and other essential tools
# Git is needed for pip installations from Git repositories if you use them.
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ttide library folder and install it using setup.py
# This assumes the unzipped folder is named ttide_py-master and is in the same directory as the Dockerfile
COPY ttide_py-master/ ./ttide_py-master/

# Change directory into the ttide folder and run the installation command
WORKDIR ttide_py-master
RUN python setup.py install

# Change back to the main app directory
WORKDIR /app

# Copy all necessary data and model files into the container
COPY TideCompiled.csv .
COPY flood_risk_model.pkl .
COPY flood_prediction_model.pkl .
COPY tide_arima_model.pkl .

# Copy the rest of your application code
COPY . .

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]

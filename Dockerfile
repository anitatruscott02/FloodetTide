# Start with a more robust base image
FROM debian:buster-slim

# Set the working directory
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Add non-free repositories to the sources list
RUN echo "deb http://deb.debian.org/debian/ stable main contrib non-free" >> /etc/apt/sources.list

# Copy the zipped ttide library and install the utility to extract it
COPY ttide_py-master.rar .
RUN apt-get clean && apt-get update
RUN apt-get install -y --no-install-recommends unrar && rm -rf /var/lib/apt/lists/*
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

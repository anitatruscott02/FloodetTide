# Use the official Python image as a base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies, including git
RUN apt-get update && apt-get install -y git

# Clone the ttide repository from the correct URL
RUN git clone https://github.com/moflaher/ttide_py.git /tmp/ttide

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python packages
# We install ttide from the local clone, and other packages from requirements.txt
RUN pip install --no-cache-dir /tmp/ttide && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY app.py .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]

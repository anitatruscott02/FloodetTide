# Use a lightweight Python 3.10 image as a base
FROM python:3.10-slim

# Set the working directory inside the container to /app
# This is where your code will live within the image
WORKDIR /app

# Copy your requirements.txt file into the container
COPY requirements.txt .

# Install the standard Python dependencies from the requirements file.
# The --no-cache-dir flag helps keep the image size small.
RUN pip install --no-cache-dir -r requirements.txt

# This is the crucial step that installs ttide directly from its GitHub repository.
# It bypasses the PyPI registry where the library is not available.
RUN pip install git+https://github.com/b-tom/ttide.git#egg=ttide

# Copy the rest of your application code (including main.py) into the container
COPY . .

# Expose port 8501, which is the default port for Streamlit apps
EXPOSE 8501

# Define the command that will run your app when the container starts
CMD ["streamlit", "run", "main.py"]

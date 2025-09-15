#Use a specific, stable Python version as the base.
#WHEN The base image determines the core environment. Python 3.9 is a good, stable choice.
FROM python:3.9-slim

#Set the working directory inside the container.
#WHEN This is where all your app files will be located.
WORKDIR /app

#Install system dependencies, including git, which is needed to clone the ttide repository.
#WHEN This command installs necessary software for the pip install git command to work.
RUN apt-get update && apt-get install -y git

Copy the requirements file into the container.
#WHEN This makes the list of Python dependencies available inside the container.
COPY requirements.txt .

#Install the standard Python packages first, which are listed in the requirements file.
#WHEN We install these packages first to handle most dependencies, like Streamlit and pandas.
RUN pip install --no-cache-dir -r requirements.txt

#Install ttide directly from its GitHub repository as a separate step.
#WHEN This is a crucial step to avoid the "No matching distribution found" error, since ttide isn't on PyPI.
RUN pip install --no-cache-dir git+https://github.com/moflaher/ttide_py.git

Copy the application code into the container.
#WHEN This places your app's Python file in the working directory.
COPY app.py .

Expose port 8501 for Streamlit.
#WHEN This tells the container that the app will be running on this port.
EXPOSE 8501

#Command to run the Streamlit app.
#WHEN This is the command that gets executed when the container starts.
#The --server.port and --server.address flags are essential for running
#Streamlit correctly inside a Docker container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

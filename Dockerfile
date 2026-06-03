# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files 
# and to ensure console output is not buffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (required for some ML libraries like XGBoost)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API using Uvicorn
# Note: Adjust the import path if your serve.py is inside src/ (e.g., src.serve:app)
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
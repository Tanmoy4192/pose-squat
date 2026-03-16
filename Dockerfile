#start with os + runtime
FROM python:3.10-slim
#run everything inside /app folder
WORKDIR /app
# install system dependencies required by opencv
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
#copy dependency file
COPY requirements.txt .
#install dependencies
RUN pip install --no-cache-dir -r requirements.txt
#copy project files
COPY . .
#start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
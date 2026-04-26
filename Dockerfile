# base image - official Python 3.10 slim image
FROM python:3.10-slim

# set working directory inside container
WORKDIR /app

# copy requirements first - Docker caches this layer
# if requirements haven't changed, Docker skips reinstalling
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the project
COPY . .

# install your package in editable mode
RUN pip install -e .

# expose port 8000 so the container can receive requests
EXPOSE 8000

# command to run when container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
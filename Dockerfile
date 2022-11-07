# Selecting the image file.
FROM python:3.9

# Working directory.
WORKDIR /code

# Copy the requirements text file (Dependencies).
COPY requirements.txt /code/requirements.txt

# Installing dependecies.
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application files.
COPY Main/api /code
COPY Main/methods /code
COPY Main/saved_models /code
COPY Main/temp /code

# Run the application.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]

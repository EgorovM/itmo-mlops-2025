# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the dependencies file to the working directory
COPY pyproject.toml pyproject.lock ./

# Install Poetry
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python && \
    . $HOME/.poetry/env && poetry config virtualenvs.create false

# Install dependencies
RUN . $HOME/.poetry/env && poetry install --no-dev

# Copy the rest of the application code to the container
COPY . .

# Run the application
CMD ["python", "src/main.py"
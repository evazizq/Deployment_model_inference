# Use an official Python image
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy only the dependency files first to leverage Docker caching
COPY pyproject.toml poetry.lock /app/

# Install dependencies without creating a virtual environment
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi

# Install pytest and pytest-mock
RUN poetry add --dev pytest pytest-mock

# Copy the rest of the project files
COPY . /app/

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app

# Run tests
RUN PYTHONPATH=/app poetry run pytest

# Command to run the inference script
CMD ["poetry", "run", "python", "src/inference.py"]
# Use an official Python image
FROM python:3.13-slim

# Install Poetry
RUN pip install poetry

# Set the working directory
WORKDIR /app

# Copy only the dependency files first to leverage Docker caching
COPY pyproject.toml poetry.lock /app/

# Install dependencies without creating a virtual environment
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi

# Install pytest
RUN poetry add --dev pytest

# Copy the rest of the project files
COPY . /app/

# Run tests
RUN poetry run pytest

# Command to run the inference script
CMD ["poetry", "run", "python", "src/inference.py"]
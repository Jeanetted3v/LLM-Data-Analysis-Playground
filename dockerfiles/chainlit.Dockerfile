FROM python:3.12-slim 

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make sure chainlit runs on port 8080 (Cloud Run default)
ENV PORT=8080

# Command to run the application
CMD ["chainlit", "run", "src.pandas_ai.chat.py", "--port", "8080", "--host", "0.0.0.0"]

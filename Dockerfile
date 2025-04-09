# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /LLMHitBigQuery-USCensus

# Install dependencies (use --no-cache to keep image small)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Set environment variables to optimize Streamlit
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_RUN_ON_SAVE=true

# Run the Streamlit app
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

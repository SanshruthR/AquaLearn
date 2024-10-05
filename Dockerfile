# Use official python image
FROM python:3.10

# Install system dependencies (for H2O & Java)
RUN apt-get update && apt-get install -y openjdk-17-jre-headless

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PATH="$JAVA_HOME/bin:$PATH" \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create a non-root user
RUN useradd -m -s /bin/bash streamlit

# Create app directory and set permissions
RUN mkdir -p /app /app/saved_models /app/.streamlit \
    && chown -R streamlit:streamlit /app

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Create Streamlit config file with XSRF protection disabled
RUN echo '[server]' > /app/.streamlit/config.toml \
    && echo 'maxUploadSize = 200' >> /app/.streamlit/config.toml \
    && echo 'enableXsrfProtection = false' >> /app/.streamlit/config.toml

# Set permissions
RUN chmod -R 755 /app \
    && chmod -R 777 /app/saved_models \
    && chown -R streamlit:streamlit /app

# Switch to non-root user
USER streamlit

# Expose the port Streamlit runs on
EXPOSE 7860

# Run Streamlit app with XSRF protection disabled
CMD ["streamlit", "run", "--server.enableXsrfProtection=false", "app.py"]
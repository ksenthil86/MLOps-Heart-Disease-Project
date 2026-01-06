# 1. Base Image: Lightweight Python environment
# Using Python slim image which is lightweight
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR $APP_HOME

# 2. Copy Files
# Copy requirements.txt first for better caching
COPY requirements.txt .

# 3. Install Dependencies
# Add Flask if not already in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install flask

# 4. Copy Application Code
# Copy the trained model
COPY models/ $APP_HOME/models/

# Copy app.py script
COPY app.py $APP_HOME

# 5. Container Run Command
# Expose port 8080 where Flask server will run
EXPOSE 8080

# Run app.py directly with Python
CMD ["python", "app.py"]
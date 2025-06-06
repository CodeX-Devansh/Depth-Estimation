# 1. Base Image
FROM python:3.10-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Copy Requirements File
COPY requirements.txt .

# 4. Install Dependencies
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# 5. Copy Application Code
COPY ./app.py .
COPY ./index.html .
# COPY ./background.jpg . # <-- Uncomment if needed

# --- Environment Variables & Writable Dirs ---
# Define cache directories within /tmp
ENV TORCH_HOME=/tmp/torch_cache
ENV HF_HOME=/tmp/hf_cache
ENV MPLCONFIGDIR=/tmp/matplotlib_cache

# Create the directories AND set MORE PERMISSIVE permissions
# Create directories and make them world-writable (777)
# This is less secure but often works around permission issues in /tmp
RUN mkdir -p /tmp/torch_cache /tmp/hf_cache /tmp/matplotlib_cache && \
    chmod -R 777 /tmp/torch_cache /tmp/hf_cache /tmp/matplotlib_cache
# Still create a non-root user for running the app itself
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser && \
    chown -R appuser:appgroup /app # Own the app code directory
# --- End Environment Variables & Dirs ---

# 6. Expose Port
EXPOSE 7860

# --- Switch to the non-root user ---
# The app will run as appuser, but the /tmp dirs are world-writable
USER appuser

# 7. Set Healthcheck (Optional but Recommended)
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s \
  CMD curl --fail http://localhost:7860/ || exit 1

# 8. Run Command
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120"]

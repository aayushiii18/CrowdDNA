FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Gradio port and bind to all interfaces
ENV GRADIO_SERVER_NAME="0.0.0.0"
EXPOSE 7860

CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT:-7860} python app.py"]

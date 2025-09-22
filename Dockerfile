# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 5000 8080

# Create startup script
RUN echo '#!/bin/bash\n\
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH\n\
# Start backend in background\n\
cd /app && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 &\n\
# Start frontend\n\
streamlit run app.py --server.port 5000 --server.address 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the application
CMD ["/app/start.sh"]

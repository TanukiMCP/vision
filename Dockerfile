FROM python:3.11-slim

# Install system dependencies for GUI automation
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxtst6 \
    libxss1 \
    libgconf-2-4 \
    libasound2 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxfixes3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcb1 \
    xvfb \
    x11vnc \
    fluxbox \
    wmctrl \
    scrot \
    && rm -rf /var/lib/apt/lists/*

# Set display environment
ENV DISPLAY=:1

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a startup script that sets up virtual display
RUN echo '#!/bin/bash\n\
# Start Xvfb\n\
Xvfb :1 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &\n\
sleep 2\n\
# Start window manager\n\
fluxbox &\n\
# Start the MCP server\n\
python server.py' > /start.sh && chmod +x /start.sh

EXPOSE 8000

# Run the startup script
CMD ["/start.sh"] 
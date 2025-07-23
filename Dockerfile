# Dockerfile for OneSun Streamlit app
FROM python:3.11-slim

# set working directory
WORKDIR /app

# prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# add src to PYTHONPATH for module imports
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# install system dependencies (if any needed for torch etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy project files
COPY . .

# set working directory for the app
WORKDIR /app/apps/main

# expose Cloud Run default port
EXPOSE 8501

CMD streamlit run main.py --server.port=${PORT:-8501} --server.address=0.0.0.0

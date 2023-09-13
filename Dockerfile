FROM python:3.10.12-slim-buster

WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip==23.2.1
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "main.py"]

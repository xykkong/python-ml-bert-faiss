FROM python:3.11-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg libavcodec-extra flac

RUN pip install --upgrade pip==23.3.2

COPY requirements.txt .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]



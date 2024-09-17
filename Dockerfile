FROM python:3.10
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY deploy.py /app/
COPY utils.py /app/
COPY rag_classes.py /app/

FROM python:3.11-alpine

RUN pip install flask prometheus_client requests
WORKDIR /app
COPY sidecar.py /app/sidecar.py
EXPOSE 9100
CMD ["python", "sidecar.py"]

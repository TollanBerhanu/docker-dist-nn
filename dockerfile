FROM python:3.9-slim
WORKDIR /app
COPY neuron.py /app/neuron.py
CMD ["python", "neuron.py"]

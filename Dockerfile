FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt requirements-ci.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ci.txt
COPY . .
CMD ["pytest", "-q"]

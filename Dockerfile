FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements-ci.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ci.txt

COPY . .

LABEL maintainer="mdnaimhassansaykat@gmail.com" \
      description="Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019"

RUN useradd -m appuser
USER appuser

CMD ["pytest", "-q"]

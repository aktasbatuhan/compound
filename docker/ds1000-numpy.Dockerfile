FROM python:3.10-slim

RUN pip install --no-cache-dir numpy==1.26.4 pandas==1.5.3 scipy==1.11.4

WORKDIR /work
COPY docker/ds1000_eval.py /usr/local/bin/ds1000_eval.py

ENTRYPOINT ["python", "/usr/local/bin/ds1000_eval.py"]

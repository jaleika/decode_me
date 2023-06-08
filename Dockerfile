FROM python:3.10.6-buster

COPY decode/api decode/api
COPY decode/preprocessing decode/preprocessing
COPY decode/params.py decode/params.py
COPY models models

COPY req_small.txt req_small.txt
COPY setup_small.py setup.py
RUN  pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install .

COPY Makefile Makefile
CMD uvicorn decode.api.fast:app --host=0.0.0.0 --port=$PORT

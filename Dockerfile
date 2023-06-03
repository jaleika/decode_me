FROM python:3.10.6-buster

COPY models/latest_model.h5 models/latest_model.h5
COPY decode/api decode/api
COPY decode/ml_logic/face_detection/main.py decode/ml_logic/face_detection/main.py
COPY decode/params.py decode/params.py
COPY req_small.txt req_small.txt
COPY setup_small.py setup.py
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r req_small.txt



COPY Makefile Makefile
CMD uvicorn decode.api.fast:app --host=0.0.0.0 --port=$PORT

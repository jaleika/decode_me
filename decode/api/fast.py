import pandas as pd
import numpy as np
from cv2 import cv2
from starlette.responses import Response

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from decode.ml_logic.emotion_detection.emotion_baseline import get_dummy_model

app = FastAPI()
# the way to load the model into memory
app.state.model =  get_dummy_model("dummy_emotion")


@app.post("/predict")
async def predict(image: UploadFile=File()):
    content = await image.read()

    nparr =  np.fromstring(content, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    y_pred = app.state.model.predict(cv2_img)
    detection = dict({0: 'angry',
                        1: 'disgust',
                        2: 'fear',
                        3: 'happy',
                        4: 'neutral',
                        5: 'sad',
                        6: 'surprise'})
    return detection[y_pred]
    #return {'status quo': f"dummy emotion model loaded prediction is: {detection[y_pred]}"}


@app.get("/")
def root():
    return { 'greeting': 'Hello Isabella, Natasha and Hayri. The FastAPI file is set up.'}

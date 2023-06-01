import pandas as pd
import numpy as np
import cv2
from starlette.responses import Response
import pickle
# Export Pipeline as pickle file

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from decode.ml_logic.emotion_detection.emotion_baseline import get_dummy_model
from decode.ml_logic.face_detection.main import get_face_detection_model

app = FastAPI()

model_face_detection = get_yolo_model()
model_emotion = get_dummy_model("dummy_emotion")

@app.get("/predict")
def predict(img):
    eq_img = histogram_equalizer(img)
    faces, face_positions = app.state.model_yolo.predict(eq_img)
    predictions = []
    for face in faces:
        predictions.append(app.state.model_emotion.predict(face))
    #model.predict()
    # return list of detected face positions (bounding box) and predictions
    return {'status quo': 'dummy emotion model loaded, no face image -> no prediction'}
#app.state.model.predict(...)
=======
# the way to load the model into memory
app.state.model =  get_dummy_model("dummy_emotion")


@app.post("/predict")
async def predict(image: UploadFile=File()):
    content = await image.read()

    nparr =  np.fromstring(content, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    expanded = (np.expand_dims(cv2_img,0))
    y_pred = app.state.model.predict(expanded)
    detection =       {0: 'angry',
                        1: 'disgust',
                        2: 'fear',
                        3: 'happy',
                        4: 'neutral',
                        5: 'sad',
                        6: 'surprise'}
    final = detection[y_pred[0]]
    return {'mood': final}
    #return {'status quo': f"dummy emotion model loaded prediction is: {detection[y_pred]}"}


@app.get("/")
def root():
    return { 'greeting': 'Hello Isabella, Natasha and Hayri. The FastAPI file is set up.'}

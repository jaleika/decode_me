import pandas as pd
from fastapi import FastAPI
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


@app.get("/")
def root():
    return { 'greeting': 'Hello Isabella, Natasha and Hayri. The FastAPI file is set up.'}

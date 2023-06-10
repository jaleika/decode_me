import pandas as pd
import numpy as np
import cv2
from starlette.responses import Response
from decode.params import *
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from decode.ml_logic.emotion_detection.emotion_baseline import get_dummy_model
from decode.ml_logic.face_detection.main import (
    get_face_detection_model,
    histogram_equalization,
)
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle

app = FastAPI()
# the way to load the model into memory
model_face_detection = get_face_detection_model()
DIR_MODELS = f"{LOCAL_MODELS_DATA_PATH}/models"
model_emotion = load_model(f"{DIR_MODELS}/Trained_InceptionResNetV2_no_resize_4")


@app.post("/predict")
async def predict(image: UploadFile = File()):
    content = await image.read()
    nparr = np.fromstring(content, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray
    with open("cv2_img.pkl", "wb") as file:
        pickle.dump(cv2_img, file)
    eq_img = histogram_equalization(cv2_img)
    with open("eq_img.pkl", "wb") as file:
        pickle.dump(eq_img, file)

    corners, key_points = model_face_detection.predict(eq_img)
    y_pred = []
    for face_corners in corners[0]:
        face_img = eq_img[
            face_corners[1] : face_corners[3],
            face_corners[0] : face_corners[2],
        ]
        face_img_resize = tf.image.resize(face_img, [150, 150])
        y_pred.append(model_emotion.predict(np.expand_dims(face_img_resize, 0)))
    print(f"y_pred is {y_pred}")
    detection = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise",
    }
    print(y_pred)
    final = [detection[np.argmax(pred)] for pred in y_pred]
    return {"mood": final, "corners": corners}  # }str(len(final))}
    # return {'status quo': f"dummy emotion model loaded prediction is: {detection[y_pred]}"}


@app.get("/")
def root():
    return {
        "greeting": "Hello Isabella, Natasha and Hayri. The FastAPI file is set up."
    }

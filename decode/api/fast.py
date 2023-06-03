import numpy as np
import cv2
from starlette.responses import Response
from decode.params import *
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from decode.ml_logic.emotion_detection.emotion_baseline import get_dummy_model
from decode.ml_logic.face_detection.main import get_face_detection_model, histogram_equalization
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle

app = FastAPI()
# the way to load the model into memory
#app.state.model_face_detection = get_face_detection_model()

DIR_MODELS = f"{LOCAL_MODELS_DATA_PATH}/models"

# put the latest model into the models-folder and rename it to 'latest_model'
app.state.model_emotion = load_model(f'{DIR_MODELS}/latest_model.h5')
with open(f'{DIR_MODELS}/model_face_detection.pkl', 'rb') as f:
    app.state.model_face_detection = pickle.load(f)
#app.state.model_face_detection = get_face_detection_model()

@app.post("/predict")
async def predict(image: UploadFile=File()):
    content = await image.read()
    nparr =  np.fromstring(content, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    eq_img = histogram_equalization(cv2_img)
    corners, key_points = app.state.model_face_detection.predict(eq_img)
    y_pred = []
    for i in range(len(corners[0])):
        face_corners = corners[0][i]
        face_img = eq_img[
                    face_corners[1] : face_corners[3],
                    face_corners[0] : face_corners[2],
                ]

        face_bw = tf.stack([face_img.mean(axis = -1), face_img.mean(axis = -1), face_img.mean(axis = -1)], axis =  -1)
        face_bw_resize = tf.image.resize(face_bw, [150,150])
        y_pred.append(app.state.model_emotion.predict(np.expand_dims(face_bw_resize,0)))
#        y_pred.append(app.state.model_emotion.predict(np.expand_dims(face_img_resize,0)))

    detection =       {0: 'angry',
                        1: 'disgust',
                        2: 'fear',
                        3: 'happy',
                        4: 'neutral',
                        5: 'sad',
                        6: 'surprise'}
    print(y_pred)
    final = [detection[np.argmax(pred)] for pred in y_pred]
    return {'mood':final, 'corners': corners}#}str(len(final))}
    #return {'status quo': f"dummy emotion model loaded prediction is: {detection[y_pred]}"}


@app.get("/")
def root():
    return { 'greeting': 'Hello Isabella, Natasha and Hayri. The FastAPI file is set up.'}

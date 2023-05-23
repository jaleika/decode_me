import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from decode.ml_logic.emotion_detection.emotion_baseline import get_dummy_model

app = FastAPI()
#app.state.model = ...

@app.get("/predict")
def predict():
    # load dummy model
    model = get_dummy_model("dummy_emotion")
    #model.predict()
    return {'status quo': 'dummy emotion model loaded, no face image -> no prediction'}
#app.state.model.predict(...)


@app.get("/")
def root():
    return { 'greeting': 'Hello Isabella, Natasha and Hayri. The FastAPI file is set up.'}

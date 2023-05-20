import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
#app.state.model = ...

@app.get("/predict")
def predict():
    return {'status quo': 'no model -> no prediction'}
#app.state.model.predict(...)


@app.get("/")
def root():
    return { 'greeting': 'Hello Isabella, Natasha and Hayri. The FastAPI file is set up.'}

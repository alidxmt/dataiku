from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd
from model.predict import *
from typing import Union

app = FastAPI()
with open ('data/model_pickle','rb') as f:
  pipeline = pickle.load(f)
@app.get('/')
def index():
    test_data = pd.read_csv('data/census_income_learn_edited.csv',skipinitialspace=True)
    pr = predict(get_pipeline(),(test_data.iloc[6:7]))
    return {'prediction': f"{'+50000' if str(pr)=='1' else '-50000'}"}

@app.get("/predict")
async def predictor(_data):
    test_data = pd.read_csv('data/census_income_learn_edited.csv',skipinitialspace=True)
    _data = (test_data.iloc[6:7])
    pr = predict(get_pipeline(),_data)
    return {'prediction': f"{'+50000' if str(pr)=='1' else '-50000'}"}

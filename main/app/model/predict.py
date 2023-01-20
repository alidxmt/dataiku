import pickle
import numpy as np
import pandas as pd




def pred_data(_data):
  '''this function get data and make it ready for pipeline
     in terms of dtype, null values and droping columns
     that we dropped when we built the model
     it return data to make a prediction
  '''
  _data = _data.replace(['?','Not in universe'], np.NaN);
  columns_to_non_numerical = ["year", "fill inc questionnaire for veteran's admin", "veterans benefits", "detailed industry recode", "detailed occupation recode"]
  for a_col in columns_to_non_numerical:
    _data[a_col] = _data[a_col].astype(str)
  _data_nul = (_data.isna().sum()*100 / len(_data))
  data_missing_columns = np.array((_data_nul[_data_nul>30]).index.tolist())
  _data = _data.drop(columns=data_missing_columns)
  X_test = _data.drop(columns='income')
  return X_test

def get_pipeline():
  with open ('data/model_pickle','rb') as f:
    return pickle.load(f)

def predict(pipeline,_data):
  X_test = pred_data(_data)
  result = pipeline.predict(X_test.iloc[0:1])[0]
  return result


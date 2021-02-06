from pydantic.types import Json
import uvicorn 
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from typing import List
import numpy as np 
import joblib 
from pydantic import BaseModel 



#load model

app = FastAPI()

model = joblib.load("myModel.sav")

@app.get('/')
def index():
    return {"API" : "Ready to call"}

class OutData(BaseModel):
    pred : float

class DataType(BaseModel):
    product : str
    calories : float 
    carbs : float 
    time : str 
    dish : str 
    heat : str 
    fat : float 
    no_ingredients : int 
    proteins : float 
    protein_class : str 
    cuisine : str 

@app.post('/predict', response_model=OutData)
def prediction(data : DataType):
    data = data.dict()  
    prd = data.get("product")
    cal = data.get("calories")
    carb = data.get("carbs")
    time = data.get("time")
    dsh = data.get("dish")
    heat = data.get("heat")
    fat = data.get("fat")
    ingrd = data.get("no_ingredients")
    prot = data.get("proteins")
    pro_clss = data.get("protein_class")
    cuisine = data.get("cuisine")
    answer = model.predict([prd, cal, carb, time, dsh, heat, fat, ingrd, prot, pro_clss, cuisine])
    answer = np.exp(answer)
    answer = OutData(pred = answer)
    answer = jsonable_encoder(answer)
    
    return answer
    
    '''
    if answer:
            return {"statusCode": 200,
                "body": {"Expected sales": answer}}
    else:
        return {"status": 404,
                "body": {"Message": "Are you sure you're using the right data ?"}}
    

    if answer < 1000:
        return {f"Sales is low with value{answer}. Ensure to increase your input"}

    else:
        return {f"Sales is high with value {answer}. You can still increase your input"}

'''
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)

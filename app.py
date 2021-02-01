import uvicorn 
from fastapi import FastAPI
import numpy as np 
import joblib 
from pydantic import BaseModel 


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

#load model

app = FastAPI()

model = joblib.load("myModel.sav")

@app.get('/')
def index():
    return {"API" : "Ready to call"}


@app.post('/predict')
def prediction(data: DataType):
    data = data.dict()
    prd = data['product']
    cal = data['calories']
    carb = data['carbs']
    time = data['time']
    dsh = data['dish']
    heat = data['heat']
    fat = data['fat']
    ingrd = data['no_ingredients']
    prot = data['proteins']
    pro_clss = data['protein_class']
    cuisine = data['cuisine']
    answer = model.predict([prd, cal, carb, time, dsh, heat, fat, ingrd, prot, pro_clss, cuisine])
    answer = np.exp(answer)

    if answer:
            return data, answer
    else:
        return {"status": 404,
                "body": {"Message": "Are you sure you're using the right data ?"}}
    
'''
    if answer < 1000:
        return {f"Sales is low with value{answer}. Ensure to increase your input"}

    else:
        return {f"Sales is high with value {answer}. You can still increase your input"}

'''
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)

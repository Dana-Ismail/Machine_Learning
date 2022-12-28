from http import HTTPStatus
from fastapi import FastAPI
from typing import Dict
import pickle
import uvicorn
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

pipeline: Pipeline
pipeline = pickle.load(open("D:/ml2/model.pkl", 'rb'))
app = FastAPI(debug=True)

@dataclass
class Input():
    name :str = None
    model_year : int = None
    color :str= None
    fuel :str= None
    origin : str= None
    bi_license :str= None
    gear_type :str= None
    bi_glass :str= None
    engine_power : float= None
    speedometer :float= None
    passenger_seat_no :int=None
    bi_payment_option :str= None
    status :str= None
    previous_owner :int= None
    bi_condition :str= None
    bi_lock_system :str= None
    bi_alert :str= None
    bi_radio :str= None
    bi_sunroof :str= None
    bi_wheels :str= None
    bi_seat_type :str= None
    bi_cushion :str= None

@dataclass  
class Output():
    prediction: int


def prediction(input:Input, pipeline):
    prediction = pipeline.predict([list(input)])[0]
    return prediction

@app.get("/")
def healthCheck():
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }

@app.post('/predict')
async def model_prediction(input :Input) -> Dict:
    dict = vars(input)
    values= list(dict.values())
    response:Output = prediction((values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8],values[9],values[10],values[11],values[12],values[13],values[14],values[15],values[16],values[17],values[18],values[19],values[20],values[21]),pipeline)
    return response

 ## JSON INPUT
# {"name": "كيا اوبتيما" ,"model_year": 2014, "color":"أبيض عاجي", "fuel":"بنزين", "origin":"خصوصي",
# "bi_license":"فلسطينية","gear_type":"اوتوماتيك", "bi_glass":"الكتروني","engine_power":2000,
# "speedometer":75000.0, "passenger_seat_no":5,"bi_payment_option":"نقدا فقط ","status":"للبيع فقط ",
# "previous_owner":1,"bi_condition":"مُكيّف","bi_lock_system":"إغلاق مركزي","bi_alert":"جهاز إنذار",
# "bi_radio":"مسجل CD","bi_sunroof":"فتحة سقف","bi_wheels":"جنطات مغنيسيوم","bi_seat_type":"فرش جلد",
# "bi_cushion":"وسادة حماية هوائية"}

if __name__ == "_main_":
    uvicorn.run("api:app", host= "127.0.0.1", port=8000, reload= True)


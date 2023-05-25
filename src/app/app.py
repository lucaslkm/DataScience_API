from fastapi import Body, FastAPI
from pycaret.regression import load_model, predict_model
import pandas as pd 
import uvicorn
app = FastAPI()

rgs_model = load_model("Final_clf_LGBM")


@app.get("/")
async def home():
    return {"Ola":"Mundo"}


@app.post("/predict")
async def home(LIMIT_BAL: str = Body(...),
 SEX: str = Body(...),
 EDUCATION: str = Body(...),
 MARRIAGE: str = Body(...),
 AGE: str = Body(...),
 PAY_1: str = Body(...),
 PAY_2: str = Body(...),
 PAY_3: str = Body(...),
 PAY_4: str = Body(...),
 PAY_5: str = Body(...),
 PAY_6: str = Body(...),
 BILL_AMT1: str = Body(...),
 BILL_AMT2: str = Body(...),
 BILL_AMT3: str = Body(...),
 BILL_AMT4: str = Body(...),
 BILL_AMT5: str = Body(...),
 BILL_AMT6: str = Body(...),
 PAY_AMT1: str = Body(...),
 PAY_AMT2: str = Body(...),
 PAY_AMT3: str = Body(...),
 PAY_AMT4: str = Body(...),
 PAY_AMT5: str = Body(...),
 PAY_AMT6: str = Body(...),
 default: str = Body(...)):
    
    data = pd.DataFrame([[
        LIMIT_BAL,
        SEX,
        EDUCATION,
        MARRIAGE,
        AGE,
        PAY_1,
        PAY_2,
        PAY_3,
        PAY_4,
        PAY_5,
        PAY_6,
        BILL_AMT1,
        BILL_AMT2,
        BILL_AMT3,
        BILL_AMT4,
        BILL_AMT5,
        BILL_AMT6,
        PAY_AMT1,
        PAY_AMT2,
        PAY_AMT3,
        PAY_AMT4,
        PAY_AMT5,
        PAY_AMT6,
        default]])

    data.columns = [
        'LIMIT_BAL',
        'SEX',
        'EDUCATION',
        'MARRIAGE',
        'AGE',
        'PAY_1',
        'PAY_2',
        'PAY_3',
        'PAY_4',
        'PAY_5',
        'PAY_6',
        'BILL_AMT1',
        'BILL_AMT2',
        'BILL_AMT3',
        'BILL_AMT4',
        'BILL_AMT5',
        'BILL_AMT6',
        'PAY_AMT1',
        'PAY_AMT2',
        'PAY_AMT3',
        'PAY_AMT4',
        'PAY_AMT5',
        'PAY_AMT6',
        'default']
    
    payer_prediction = predict_model(rgs_model, data = data)

    return {"Predicao do valor do contrato": payer_prediction["LIMIT_BAL"]}

if __name__ =="__main__":
    uvicorn.run("app:app", host='0.0.0.0')
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the trained model
with open('device_price_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

app = FastAPI()

class DeviceFeatures(BaseModel):
    battery_power: int
    blue: int
    clock_speed: float
    dual_sim: int
    fc: int
    four_g: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    sc_h: int
    sc_w: int
    talk_time: int
    three_g: int
    touch_screen: int
    wifi: int

@app.post("/predict")
def predict_price(features: DeviceFeatures):
    data = np.array([[
        features.battery_power,
        features.blue,
        features.clock_speed,
        features.dual_sim,
        features.fc,
        features.four_g,
        features.int_memory,
        features.m_dep,
        features.mobile_wt,
        features.n_cores,
        features.pc,
        features.px_height,
        features.px_width,
        features.ram,
        features.sc_h,
        features.sc_w,
        features.talk_time,
        features.three_g,
        features.touch_screen,
        features.wifi
    ]])

    prediction = best_model.predict(data)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

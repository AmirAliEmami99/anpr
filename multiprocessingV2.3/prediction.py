import tensorflow as tf
def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    print("Model loaded")
    return model
model = load_model()

import uvicorn
from fastapi import FastAPI, File, UploadFile
from application.components import predict, read_imagefile
app = FastAPI()
@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)
    return prediction
@app.post("/api/covid-symptom-check")
def check_risk(symptom: Symptom):
    return symptom_check.get_risk_level(symptom)
if __name__ == "__main__":
    uvicorn.run(app, debug=True)
from fastapi import FastAPI, File, UploadFile
from fer_model import predict_fer
import uvicorn
import shutil
import base64
import json
import os


app = FastAPI()


@app.post("/api/upload_image")
async def upload_image(image: UploadFile = File(...)):
    with open("image.png", "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    success = predict_fer.detect_face("image.png")
    if success == "successful":
        emotion, value = predict_fer.predict_facial_expression("image.png")
        os.remove("image.png")
        resp = {"data": {"emotion": emotion, "value": value}}
        return json.dumps(resp)
    else:
        os.remove("image.png")
        resp = {"error": {"message": success}}
        return json.dumps(resp)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True, log_level="info")

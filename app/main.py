from fastapi import FastAPI, File, UploadFile, status, HTTPException
from fer_model import predict_fer, utility_functions

import filetype
import uvicorn
import shutil
import os

app = FastAPI()


@app.post("/api/upload_image", status_code=status.HTTP_200_OK)
async def upload_image(image: UploadFile = File(...)):
    with open("image.png", "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    if not filetype.is_image("image.png"):
        os.remove("image.png")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload an image file format.",
        )
    try:
        success = utility_functions.detect_face("image.png")
        emotion, value = predict_fer.predict_facial_expression("image.png")
        return {"emotion": emotion, "prediction_value": value}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    finally:
        os.remove("image.png")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True, log_level="info")

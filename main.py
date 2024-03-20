
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
app = FastAPI()


endpoint= "http://13.48.136.54:8000/api/api-code/"
headers = {"Authorization": "b944f209-b152-4a05-9d13-b42194c403db"}
print(requests.post(endpoint, headers=headers).json())
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
origins =[
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
]
app.add_middleware(
CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
saved_model_path = "../model/17.keras"
# saved_model_path = "../model/keras_model.h5"
MODEL = tf.keras.models.load_model(saved_model_path)
#MODEL=tf.keras.layers.TFSMLayer(saved_model_path, call_endpoint='saved_model.pb')
CLASS_NAMES = ["FU-nail-fungus", "FU-ringworm","healthy"]

@app.get("/ping")
async def ping():
    return "Working"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=800)
#uvicorn.run(app, host='localhost', port=8000)
#http://13.48.136.54:8000/api/api-code/
from fastapi import FastAPI, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_models/0.01")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get("/ping")
async def ping():
    return "Hello, I'm alive."

def process_image_data(data) -> np.ndarray:
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception as e:
        print(f"Error in process_image_data: {e}")
        return None

def predict_class_and_confidence(image: np.ndarray) -> dict:
    try:
        img_batch = np.expand_dims(image, 0)
        prediction = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))

        return {
            'class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        print(f"Error in predict_class_and_confidence: {e}")
        return {
            'error': f'Failed to make a prediction. Details: {str(e)}'
        }

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        image_data = await file.read()
        image = process_image_data(image_data)

        if image is not None:
            result = predict_class_and_confidence(image)
            print("Prediction Result:", result)
            return result
        else:
            return {'error': 'Failed to process the image.'}

    except Exception as e:
        print(f"Error in predict: {e}")
        return {'error': f'Failed to make a prediction. Details: {str(e)}'}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8060)

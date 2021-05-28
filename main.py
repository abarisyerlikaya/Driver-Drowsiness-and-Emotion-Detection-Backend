import cv2
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}
cascade_classifier = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')
emotion_detector = tf.keras.models.load_model("assets/emotion.h5", compile=True)
resize_height = 640
resize_width = 360
x = ["*"]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=x, allow_methods=x, allow_headers=x, allow_credentials=True)


@app.get("/")
async def root():
    return {"message": "Echo"}


@app.post("/classify-emotion/")
async def create_item(item: Request):
    try:
        request = await item.json()
        gray_frame = np.array(request['data']).astype('uint8').reshape((resize_height, resize_width))
        frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        faces = cascade_classifier.detectMultiScale(gray_frame)  # , scaleFactor=1.3, minNeighbors=5

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = emotion_detector.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                return {"class": emotion_dict[max_index]}
        else:
            return {"class": ""}
    except:
        return {"class": ""}


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.1.149", port=8080)

import dlib
import cv2
import numpy as np
import tensorflow as tf
import uvicorn
# from matplotlib import pyplot as plt
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware as Cmw

emotion_dict = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}

emotion_detector = tf.keras.models.load_model("assets/emotion.h5", compile=True)
drowsiness_detector = tf.keras.models.load_model("assets/drowsiness.h5", compile=True)
cascade_classifier = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')
shape_predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")
face_detector = dlib.get_frontal_face_detector()

resize_height = 480
resize_width = 480

app = FastAPI()
app.add_middleware(Cmw, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)


@app.get("/")
async def root():
    return {"message": "Echo"}


@app.post("/classify/")
async def create_item(item: Request):
    try:
        request = await item.json()
        gray_frame_org = np.fromstring(request['data'], dtype='uint8', sep=',').reshape((resize_height, resize_width))
        gray_frame = cv2.copyMakeBorder(gray_frame_org, 0, 0, 80, 80, cv2.BORDER_CONSTANT)

        # Drowsiness Detection
        mixed = []
        faces = face_detector(gray_frame)
        if len(faces) > 0:
            for face in faces:
                landmarks = shape_predictor(gray_frame, face)
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    mixed.append(x)
                    mixed.append(y)
            if len(mixed) != 0:
                mixed = np.array(mixed)
                mixed = np.expand_dims(mixed, axis=1)
                mixed = np.transpose(mixed)
                mixed = np.expand_dims(mixed, axis=0)
                prediction = drowsiness_detector.predict(mixed)
                print(prediction[0][0][0])
                if prediction[0][0][0] > 0.5:
                    return {"class": "drowsy"}

        # Emotion Detection
        faces = cascade_classifier.detectMultiScale(gray_frame_org)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray_frame_org[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = emotion_detector.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                return {"class": emotion_dict[max_index]}
        else:
            return {"class": "nf"}

    except Exception as e:
        print(e)
        return {"class": "nf"}


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.1.149", port=8080)

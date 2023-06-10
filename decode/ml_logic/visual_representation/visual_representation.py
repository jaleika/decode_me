import cv2
from decode.api.fast import predict
import numpy as np


def merge_rectangles(img, res):
    detection = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise",
    }
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    corners = res["corners"]
    mood = res["final"]
    for index, face_corners in enumerate(corners[0]):
        top_left = face_corners[0:2]
        top_left2 = [face_corners[0], face_corners[1] - 14]
        right_bottom = face_corners[2:]
        if detection[np.argmax(mood)] == "angry":
            color = (255, 0, 0)
        if detection[np.argmax(mood)] == "disgust":
            color = (0, 255, 0)
        if detection[np.argmax(mood)] == "fear":
            color = (0, 0, 0)
        if detection[np.argmax(mood)] == "happy":
            color = (7, 223, 255)
        if detection[np.argmax(mood)] == "neutral":
            color = (255, 255, 255)
        if detection[np.argmax(mood)] == "sad":
            color = (111, 111, 111)
        if detection[np.argmax(mood)] == "surprise":
            color = (255, 128, 0)
        cv2.rectangle(img2, top_left, right_bottom, color, int(img2.shape[1] / 512))
        cv2.putText(
            img2,
            detection[np.argmax(mood)],
            top_left2,
            cv2.FONT_HERSHEY_SIMPLEX,
            int(img2.shape[1] / 256),
            color,
            int(img2.shape[1] / 512),
        )
    return img2

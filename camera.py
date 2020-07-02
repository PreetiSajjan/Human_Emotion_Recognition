import cv2
from model import FacialExpressionModel
import numpy as np

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
out = cv2.VideoWriter('Emotion Detector.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (626, 626))


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture('emotion_dataset/testvideo.mp4')
        self.video.set(cv2.CAP_PROP_FPS, 30)

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def get_color(self, pred):
        if pred is 'Happy':
            col = (0, 210, 0)
        elif pred is 'Angry':
            col = (0, 0, 164)
        elif pred is 'Disgust':
            col = (0, 147, 136)
        elif pred is 'Fear':
            col = (0, 73, 255)
        elif pred is 'Sad':
            col = (115, 117, 57)
        elif pred is 'Surprise':
            col = (0, 225, 255)
        else:
            col = (51, 52, 0)
        return col

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        while True:
            # Grab a single frame of video
            ret, fr = self.video.read()

            if ret:
                gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray_fr, 1.3, 5)

                for (x, y, w, h) in faces:
                    fc = gray_fr[y:y + h, x:x + w]

                    roi = cv2.resize(fc, (48, 48))
                    pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                    col = self.get_color(pred)
                    cv2.putText(fr, pred, (x, y-10), font, 1.5, col, 3)
                    cv2.rectangle(fr, (x, y), (x + w, y + h), col, 3)

                cv2.imshow('Emotion Detector', fr)
                out.write(fr)
                cv2.waitKey(10)

            else:
                break


if __name__ == '__main__':
    video = VideoCamera()
    video.get_frame()
    out.release()

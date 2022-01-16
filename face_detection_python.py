import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv.CascadeClassifier(cascPath)
model = load_model('model/bestmodel (dev model).hdf5')
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
video_capture = cv.VideoCapture(1)
i = 0
while True:
    ret, frame = video_capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(80, 80),
        flags=cv.CASCADE_SCALE_IMAGE
    )
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        w_r = int(0.25 * w) * 0
        h_r = int(0.25 * h) * 0
        face = cv.resize(gray[y-h_r:y+h+h_r, x-w_r:x+w+w_r], (48, 48)) / 255.0
        face = face.reshape([-1, 48, 48, 1])
        preds = model.predict(face).ravel()
        lbl = np.argmax(preds)
        cv.putText(frame, f'{classes[lbl]}: {(preds[lbl] * 100):.2f}%', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, .8, 255, 1, cv.LINE_AA)
    # End For
    i += 1
    # cv.imwrite(f'faces/{i}.jpg', frame)
    cv.imshow('camera', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    # End if
# End While

video_capture.release()
cv.destroyAllWindows()
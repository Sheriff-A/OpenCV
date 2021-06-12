import cv2 as cv
from deepface import DeepFace

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_width, frame_height = 640, 480
cap = cv.VideoCapture(0)
cap.set(3, frame_width)
cap.set(3, frame_height)

while True:
    _, frame = cap.read()

    # Detect Face
    face = face_cascade.detectMultiScale(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), 1.3, 5)
    for x, y, w, h in face:

        # Need to find a way to speed this up... Takes wayyyy too long
        # Gonna attempt using CV + CUDA
        predictions = DeepFace.analyze(frame, actions=['emotion'])
        # print(predictions)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv.putText(frame, 'Emotion:' + str(predictions['dominant_emotion']), (x, y + h + 20),
                   cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 1)
        # cv.putText(frame, 'Race:' + str(predictions['dominant_race']), (x, y + h + 40),
        #            cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 1)
        # cv.putText(frame, 'Gender:' + str(predictions['gender']), (x, y + h + 60),
        #            cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 1)
        # cv.putText(frame, 'Age:' + str(predictions['age']), (x, y + h + 80),
        #            cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 1)

    cv.imshow('Frame', cv.flip(frame, 1))
    if cv.waitKey(5) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

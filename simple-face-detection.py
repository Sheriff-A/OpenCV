import cv2 as cv

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()

    # Detect Face
    input_data = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(input_data, 1.3, 5)
    for x, y, w, h in face:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

        # Detect Eyes
        selected_face = input_data[y:y+h, x:x+w]
        selected_face_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(selected_face, 1.3, 5)
        for ex, ey, ew, eh in eyes:
            cv.rectangle(selected_face_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

    cv.imshow('Frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
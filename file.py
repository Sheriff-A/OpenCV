import cv2 as cv
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(3, frameHeight)


def empty(a):
    pass


cv.namedWindow('Parameters')
cv.resizeWindow('Parameters', frameWidth, frameHeight // 2)
cv.createTrackbar('Threshold1', 'Parameters', 20, 255, empty)
cv.createTrackbar('Threshold2', 'Parameters', 20, 255, empty)
cv.createTrackbar('Area', 'Parameters', 2000, 30000, empty)


def get_contours(img, img_contour):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        area_min = cv.getTrackbarPos('Area', 'Parameters')
        if area > area_min:
            cv.drawContours(img_contour, cnt, -1, (255, 0, 255), 7)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            x, y, w, h = cv.boundingRect(approx)
            cv.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv.putText(img_contour, 'Points: ' + str(len(approx)), (x + w + 20, y + 20),
                       cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            cv.putText(img_contour, 'Area: ' + str(int(area)), (x + w + 20, y + 45),
                       cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)


while True:
    _, frame = cap.read()
    # frame = cv.flip(frame, -1)  # Vertical Flip
    contour = frame.copy()

    blur = cv.GaussianBlur(frame, (7, 7), 1)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

    threshold1 = cv.getTrackbarPos('Threshold1', 'Parameters')
    threshold2 = cv.getTrackbarPos('Threshold2', 'Parameters')
    canny = cv.Canny(gray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    dil = cv.dilate(canny, kernel, iterations=1)
    get_contours(dil, contour)

    frame = np.hstack([frame, cv.cvtColor(dil, cv.COLOR_GRAY2BGR), contour])
    cv.imshow('Frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

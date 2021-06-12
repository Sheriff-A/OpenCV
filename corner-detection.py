import numpy as np
import cv2 as cv


seed = 12
np.random.seed(seed)

img = cv.imread('assets\chessboard.png')
img = cv.resize(img, (0, 0), fx=0.75, fy=0.75)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray, 100, 0.01, 10)
# print(corners)
corners = np.int0(corners)  # Converts Floats to Ints
# print(corners)

# img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
for corner in corners:
    x, y = corner.ravel()
    cv.circle(img, (x, y), 5, (0, 255, 0), -1)

for i in range(len(corners)):
    for j in range(i+1, len(corners)):
        c1 = tuple(corners[i].ravel())
        c2 = tuple(corners[j].ravel())
        color = tuple(map(lambda x: int(x), np.random.randint(0, 255, 3)))
        cv.line(img, c1, c2, color, 1)

cv.imshow('Corner Detection', img)
cv.waitKey(0)
cv.destroyAllWindows()

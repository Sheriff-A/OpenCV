import cv2 as cv
import numpy as np

img = cv.imread('assets/soccer_practice.jpg', 0)
template = cv.imread('assets/ball.PNG', 0)

h, w = template.shape
methods = [cv.TM_CCOEFF,
           cv.TM_CCOEFF_NORMED,
           cv.TM_CCORR,
           cv.TM_CCORR_NORMED,
           cv.TM_SQDIFF,
           cv.TM_SQDIFF_NORMED]

for idx, method in enumerate(methods):
    img2 = img.copy()
    result = cv.matchTemplate(img2, template, method)
    _, _, min_loc, max_loc = cv.minMaxLoc(result)
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc
    bottom_right = (location[0] + w, location[1] + h)
    cv.rectangle(img2, location, bottom_right, 255, 5)
    cv.imshow('Result', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()



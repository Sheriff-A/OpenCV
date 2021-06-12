import cv2 as cv
import numpy as np

seed = 12
np.random.seed(seed)


# Blank function for sliders
def empty(a):
    pass


# Create the sliders
cv.namedWindow('HSV')
cv.resizeWindow('HSV', 640, 240)
cv.createTrackbar('HUE Min', 'HSV', 0, 179, empty)
cv.createTrackbar('HUE Max', 'HSV', 179, 179, empty)
cv.createTrackbar('SAT Min', 'HSV', 0, 255, empty)
cv.createTrackbar('SAT Max', 'HSV', 255, 255, empty)
cv.createTrackbar('VAL Min', 'HSV', 0, 255, empty)
cv.createTrackbar('VAL Max', 'HSV', 255, 255, empty)

# Captures Feed From Video Camera
cap = cv.VideoCapture(0)
while True:
    # Read Frame by Frame
    _, frame = cap.read()
    # Convert Frame to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Get Slider Values
    h_min = cv.getTrackbarPos('HUE Min', 'HSV')
    h_max = cv.getTrackbarPos('HUE Max', 'HSV')
    s_min = cv.getTrackbarPos('SAT Min', 'HSV')
    s_max = cv.getTrackbarPos('SAT Max', 'HSV')
    v_min = cv.getTrackbarPos('VAL Min', 'HSV')
    v_max = cv.getTrackbarPos('VAL Max', 'HSV')

    # Create Color Bounds
    lower_red = np.array([h_min, s_min, v_min])
    upper_red = np.array([h_max, s_max, v_max])

    # Create and Apply Mask
    mask = cv.inRange(hsv, lower_red, upper_red)
    res = cv.bitwise_and(frame, frame, mask=mask)

    # Convert Mask
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    combined = np.hstack([frame, mask, res])
    cv.imshow('Live Color Masking', combined)

    if cv.waitKey(1) == ord('q'):
        break
    if cv.waitKey(1) == ord('s'):
        cv.imwrite('saves/original.png', frame)
        cv.imwrite('saves/mask.png', mask)
        cv.imwrite('saves/result.png', res)
        print('Files Saved!')

# Release the Camera and Close All Windows
cap.release()
cv.destroyAllWindows()
print('Bye!')

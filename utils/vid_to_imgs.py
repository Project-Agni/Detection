import os

import cv2

# cam = cv2.VideoCapture("../datasets/realtime/fire.mp4")
# save_folder = "../datasets/realtime/fire"

cam = cv2.VideoCapture("../datasets/realtime/nofire.mp4")
save_folder = "../datasets/realtime/nofire"

try:
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
except OSError:
    print("Error: Creating directory of data")

currentframe = 0

while True:
    ret, frame = cam.read()

    if ret:
        if not currentframe % 50:
            name = f"{save_folder}/frame" + str(currentframe) + ".jpg"
            print("Creating..." + name)
            cv2.imwrite(name, frame)
    else:
        break
    currentframe += 1

cam.release()
cv2.destroyAllWindows()

import cv2
from random import randrange
trainedData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#img = cv2.imread("my-photo.jpg")
webcam = cv2.VideoCapture(0)
#key = cv2.waitKey(1)

while True:
    succeful, frame = webcam.read()

    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faceCoordinate = trainedData.detectMultiScale(grayscale_img)

    for (x, y, w, h) in faceCoordinate:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
    cv2.imshow("my-photo", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break


webcam.release()
    


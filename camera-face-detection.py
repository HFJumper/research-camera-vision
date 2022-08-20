import cv2

cap = cv2.VideoCapture(0)

# https://github.com/opencv/opencv/tree/master/data/haarcascades

classfier = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32,32))

    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destoryAllWindows()
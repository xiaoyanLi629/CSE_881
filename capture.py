import cv2
#import opencv-python

video = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

a = 0
while True:
    a = a+1
    check, frame = video.read()

    # print(check)
    # print(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        # flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # print("X:", x)
        # print("Y:", y)
        # print("W:", w)
        # print("H:", h)
        face = frame[y+3:y+h-3, x+3:x+w-3]
        face = cv2.resize(face, (300, 300))
        cv2.imshow("gray_face", face)
        # key = cv2.waitKey(1)
        print(face.shape[0] * face.shape[1])

        font = cv2.FONT_HERSHEY_SIMPLEX
        upRightCornerOfText = (x+w, y)

        fontScale = 1
        fontColor = (255, 0, 0)
        lineType = 2
        if w*h >= 300*300:
            cv2.putText(frame, 'Angry',
                        upRightCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

        if w*h <= 300*300:
            cv2.putText(frame, 'Happy',
                    upRightCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()

cv2.destroyAllWindows()

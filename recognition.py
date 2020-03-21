import sys
import cv2

#Loading the haar cascade
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

#starting the webcam
video_capture = cv2.VideoCapture(0)

#the image to put on top of the face
s_img = cv2.imread("poop.png", -1)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #face detection
    faces = faceCascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        #draw rectangle around detected face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
        #resizing the overlaying image to the size of the face
        res_img = cv2.resize(s_img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        
        #putting the new image on top of the face without its background
        y1, y2 = y, y + res_img.shape[0]
        x1, x2 = x, x + res_img.shape[1]

        alpha_s = res_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * res_img[:, :, c] +
                                    alpha_l * frame[y1:y2, x1:x2, c])


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows() 


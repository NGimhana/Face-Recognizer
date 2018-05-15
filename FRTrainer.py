import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)
count=1;
path = 'training-data/s3/'

#there is also a more accurate but slow Haar classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
        
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)	
    	cv2.imwrite(os.path.join(path,str(count)+".jpg"),gray[y:y+w, x:x+h])

    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    count=count+1		

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

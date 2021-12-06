import cv2 as cv
import numpy as np


def humanDetection():
    cap = cv.VideoCapture('in.avi')
    human_cascade = cv.CascadeClassifier('haarcascade_fullbody.xml')

    # Hog descriptor accuracy
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    #involves stability of values
    previous_count = 0
    itr = 0
    ct =0
    previous_count_ct = 0

    while True:
        ct = 0
        itr += 1
        n_humans = 0

        ret,frame = cap.read()
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        humans = human_cascade.detectMultiScale(gray,1.9,1)
        boxes, weights = hog.detectMultiScale(gray, winStride=(4, 4), scale = 1.03)
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        ct = len(boxes)
        n_humans = len(humans)

        if(len(boxes)> len(humans)):
            for (xA, yA, xB, yB) in boxes:
                # display the detected boxes in the colour picture
                cv.rectangle(frame, (xA, yA), (xB, yB),
                              (0, 255, 0), 2)
        else:
            for (x,y,w,h) in humans:
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


        if previous_count < n_humans:
            previous_count = n_humans
        if previous_count_ct < ct:
            previous_count_ct = ct

        if (itr == 25):
            previous_count = n_humans
            previous_count_ct = ct
            itr = 0

        final_count = 0
        if (previous_count_ct > previous_count):
            final_count = previous_count_ct
        else:
            final_count = previous_count

        cv.putText(frame,str(final_count), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv.imshow('frame',frame)
        if cv.waitKey(50) ==ord('q'): # Press q to exit
            print("Programm ended")
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    humanDetection()
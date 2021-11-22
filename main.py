import cv2 as cv


def humanDetection():
    cap = cv.VideoCapture('in.avi')
    human_cascade = cv.CascadeClassifier('haarcascade_fullbody.xml')

    while True:
        n_humans = 0
        ret,frame = cap.read()
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        humans = human_cascade.detectMultiScale(gray,1.9,1)

        for (x,y,w,h) in humans:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            n_humans += 1

        cv.putText(frame,str(n_humans), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv.imshow('frame',frame)
        if cv.waitKey(50) ==ord('q'): # Press q to exit
            print("Programm ended")
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    humanDetection()
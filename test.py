import cv2 as cv

# def motionDetection():
#     cap = cv.VideoCapture(0)
#     ret, frame1 = cap.read()
#     ret, frame2 = cap.read()
#
#     while cap.isOpened():
#         diff = cv.absdiff(frame1,frame2)
#         diff_gray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
#         blur = cv.GaussianBlur(diff_gray,(5,5),0)
#         _, thresh = cv.threshold(blur,20,255,cv.THRESH_BINARY)
#         dilated = cv.dilate(thresh,None,iterations =3)
#         contours, _ = cv.findContours(
#             dilated,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             (x,y,w,h) = cv.boundingRect(contour)
#             if cv.contourArea(contour) <900:
#                 continue
#             cv.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
#             cv.putText(frame1,"Pedestrain ()".format("Movement"),(10,20),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
#
#         cv.imshow("Video",frame1)
#         frame1 = frame2
#         ret, frame2 = cap.read()
#         if cv.waitKey(50) == ord('q'):
#             break
#     cap.release()
#     cv.destroyAllWindows()

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

        cv.putText(frame,str(n_humans), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv.imshow('frame',frame)
        if cv.waitKey(50) ==ord('q'):
            break
            print("Programm ended")
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    humanDetection()



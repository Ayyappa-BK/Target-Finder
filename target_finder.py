import argparse as argp
import cv2 as cv

ap=argp.ArgumentParser()
ap.add_argument("-v","--video", help="path to the video file")
args=vars(ap.parse_args())

camVideo=cv.VideoCapture(args["video"])

while True:
    (grabbed, frame)=camVideo.read()
    status="No Target in sight"

    if not grabbed:
        break

    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred=cv.GaussianBlur(gray,(7,7),0)
    edged=cv.Canny(blurred,50,150)

    (cnts, _)=cv.findContours(edged.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)


for cnt in cnts:
    approx=cv.approxPolyDP(cnt,0.01*cv.arcLength(cnt,True),True)

    if len(approx)==5:
        cv.drawContours(frame, [approx], -1, (0,0,255),4)
        status="Targets on sight!"

        cv.putText(frame, status,(20,30),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        cv.imshow("Frame ",frame)
        key=cv.waitKey(1) & 0xFF

        if key==ord("s"):
            break

camVideo.release()
cv.destroyAllWindows()

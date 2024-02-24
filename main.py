import math

from ultralytics import  YOLO
import cv2
import cvzone
model = YOLO("ppe.pt")
classNames = ['Goggles', 'boots', 'gloves', 'helmet', 'mask', 'vest']



# cap = cv2.VideoCapture(0)
# taking the video as an input
cap = cv2.VideoCapture("video.mp4")
# setting up dimensions
cap.set(3,640)
cap.set(4,720)
if cap is None:
    print("error loading the camera")

while True:
    # reading the image
    success, img = cap.read()
    print(success)
    # give img as an imput to the model
    results = model(img, stream=True)
    for r in results:
        # making the boxes around the object that get detected
        boxes = r.boxes
        for box in boxes:
            # bounding box there corrdinates
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # print( x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            w, h = x2-x1,y2-y1
            # bbox = int(x1),int(y1),int(w),int(h)
            # making a line outside the box

            conf = math.floor((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = classNames[cls]
            # if current_class == "person" and conf > 0.3:
            cvzone.cornerRect(img,(x1,y1,w,h))
            # calculating the confidence value of the object that get detected
            # confidence
            conf = math.floor((box.conf[0]*100))/100
            # the max is for making sure that the confidence number stays within the frame
            cvzone.putTextRect(img,f'{conf}',(max(x1,0),max(35,y1)))
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]},{conf}', (max(x1, 0), max(35, y1)))


    cv2.imshow("Image",img)
    cv2.waitKey(1)


cv2.waitKey(0)
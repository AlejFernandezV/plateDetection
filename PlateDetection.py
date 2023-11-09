import cv2
from ultralytics import YOLO

model = YOLO('BestTrain/weights/best.pt')

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('Videos/v3.mp4')

while cap.isOpened():

    success, frame = cap.read()
    if success:

        results = model(frame, imgsz=640)
        annotated_frame = results[0].plot()

        cv2.imshow("Plates", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break    
    
cap.release()
cv2.destroyAllWindows()

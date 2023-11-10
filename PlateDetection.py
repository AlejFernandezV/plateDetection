import cv2
from ultralytics import YOLO

modelPlates = YOLO('BestTrain/weights/plates/best.pt')
modelLN = YOLO('BestTrain/weights/letters_numbers/bestFull30.pt')

cap = cv2.VideoCapture(0)
ctexto = ""
# cap = cv2.VideoCapture('Videos/v3.mp4')

while cap.isOpened():

    success, frame = cap.read()
    if success:

        resultsPlates = modelPlates(frame, imgsz=640)
        
        for result in resultsPlates.xyxy[0]:
            x, y, w, h = map(int, result[:4])
            roi = frame[y:h, x:w]

            resultsLN = modelLN(roi, imgsz=640)
            letras_numeros = resultsLN[0].plot()

            cv2.imshow("Letters and Numbers", letras_numeros)

        placas = resultsPlates[0].plot()

        cv2.imshow("Plates", placas)

        tecla = cv2.waitKey(1)

        if tecla == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

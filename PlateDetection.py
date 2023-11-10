import cv2
from ultralytics import YOLO

modelPlates = YOLO('BestTrain/weights/plates/best.pt')
modelLN = YOLO('BestTrain/weights/letters_numbers/BestF.pt')

cap = cv2.VideoCapture(0)
ctexto = ""
# cap = cv2.VideoCapture('Videos/v3.mp4')

while cap.isOpened():

    success, frame = cap.read()
    if success:

        resultsPlates = modelPlates(frame, imgsz=640)

        # Accede a un elemento de la lista resultsPlates
        resultsPlates = resultsPlates[0]

        # Accede al atributo boxes del objeto Results
        boxes = resultsPlates.boxes

        for box in boxes:
            # Verifica si el objeto box tiene un atributo xyxy
            if box is not None:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                roi = frame[y1:y2, x1:x2]
                resultletras = modelLN(roi, imgsz=640)
                letras = resultletras[0].plot(conf=None)
                h, w, _ = letras.shape
                
                frame[y1:y1 + h, x1:x1 + w] = letras
                
        # Dibuja las placas detectadas en la imagen
        placas = resultsPlates.plot(conf=None).astype('uint8')
        h,w,_ = placas.shape
        frame[:h, :w] = placas
        
        cv2.imshow("Resultados", placas)

        tecla = cv2.waitKey(1)

        if tecla == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

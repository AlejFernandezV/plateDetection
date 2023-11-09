import cv2
import pytesseract

# Cargar la imagen
image = cv2.imread('ejemplo.png')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar thresholding para binarizar la imagen
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Aplicar OCR usando Pytesseract
text = pytesseract.image_to_string(thresh, config='--psm 11')

# Imprimir el texto detectado
print(text)
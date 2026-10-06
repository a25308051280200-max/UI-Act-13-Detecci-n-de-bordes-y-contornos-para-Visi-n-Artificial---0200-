## Alejandra Rivera Valenzuela NC 0200
## PELICANO
## Ejemplo 1 (Del 1 al 30)
import os
import cv2
import numpy as np

# 1. Cargar la imagen de entrada
ruta_imagen = 'pelicano-0200.webp'

if not os.path.exists(ruta_imagen):
    print(f"Error: El archivo '{ruta_imagen}' no se encuentra en el directorio actual.")
    print("Asegúrate de que la imagen esté en la misma carpeta que este script.")
    exit()

imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print(f"Error: No se pudo cargar la imagen '{ruta_imagen}'. Verifica que el formato sea soportado por OpenCV.")
    exit()

# Redimensionar opcionalmente para trabajar mejor la visualización
imagen = cv2.resize(imagen, (500, 500))

# ==========================================
# EJEMPLO 1: Filtros del 1 al 30 (Básicos)
# ==========================================

# 1. Escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# 2. Suavizado / Desenfoque Gaussiano (Blur)
desenfoque = cv2.GaussianBlur(imagen, (15, 15), 0)

# 3. Detección de bordes con Canny
bordes = cv2.Canny(gris, 100, 200)

# 4. Umbralización (Binarización Otsu)
_, umbral = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# ==========================================
# EJEMPLO 4: Filtros del 31 al 60 (Avanzados)
# ==========================================

# 1. Detección de Bordes Sobel (Gradientes X e Y)
sobelx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=5)
sobel_combinado = cv2.magnitude(sobelx, sobely)
sobel_combinado = np.uint8(sobel_combinado)

# 2. Operación Morfológica: Dilatación y Erosión
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(umbral, kernel, iterations=1)
dilatacion = cv2.dilate(umbral, kernel, iterations=1)

# 3. Ajuste de Contraste y Brillo (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
igualada = clahe.apply(gris)


# ==========================================
# VISUALIZACIÓN DE RESULTADOS
# ==========================================

# Muestra de resultados del Ejemplo 1 con el nombre pelicano-0200 en las ventanas
cv2.imshow('pelicano-0200 - Original', imagen)
cv2.imshow('pelicano-0200 - Ejemplo 1 (Grises)', gris)
cv2.imshow('pelicano-0200 - Ejemplo 1 (Desenfoque)', desenfoque)
cv2.imshow('pelicano-0200 - Ejemplo 1 (Bordes Canny)', bordes)

# Muestra de resultados del Ejemplo 4 con el nombre pelicano-0200 en las ventanas
cv2.imshow('pelicano-0200 - Ejemplo 4 (Sobel)', sobel_combinado)
cv2.imshow('pelicano-0200 - Ejemplo 4 (Dilatacion)', dilatacion)
cv2.imshow('pelicano-0200 - Ejemplo 4 (CLAHE Contraste)', igualada)

print("Presiona cualquier tecla en una ventana de imagen para cerrar...")
cv2.waitKey(0)
cv2.destroyAllWindows()
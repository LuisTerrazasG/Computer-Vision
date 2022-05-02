import cv2 as cv
import numpy as np

#Captura de video
cap = cv.VideoCapture(0)

def nothing():
    pass

#Creacion de barras para cambiar valores de canny
cv.namedWindow("Trackbars")
cv.createTrackbar("Lower", "Trackbars",120,255,nothing)
cv.createTrackbar("Upper", "Trackbars",120,255,nothing)

while True:
    #Captura de frame del video
    _, frame = cap.read()
    frame = cv.resize(frame,(500,500))

    l = cv.getTrackbarPos("Lower","Trackbars")
    u = cv.getTrackbarPos("Upper", "Trackbars")

    #Creacion de imagenes en grises
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    imgBlur = cv.GaussianBlur(gray,(3,3),0)
    canny = cv.Canny(imgBlur,l,u)

    #Cracion de contornos
    contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    gray = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
    #Dibujar contornos
    cv.drawContours(gray,contours,-1,(255,0,0),3)
    #Mostrar imagenes
    cv.imshow("Camara", frame)
    cv.imshow("Canny", canny)
    cv.imshow("Gray color contours", gray)

    #Reconocimeinto de lineas Hough para segunda imagen en grises
    edges = cv.Canny(gray2,l,u,apertureSize=3)
    lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100, maxLineGap=10)

    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(gray2,(x1,y1),(x2,y2),(0,255,0),2)

    cv.imshow("Gray lines", gray2)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

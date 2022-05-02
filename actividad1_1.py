import cv2 as cv
import numpy as np

#Loop para lectura de imagenes de edificios
for i in range(5):
    #Lectura de imagnes y cracion de imagenes en grises, lineas y contornos
    img = cv.imread(str(f'Images_Building/img{i}.jpg'))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray,50,150,apertureSize=3)
    lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100, maxLineGap=10)

    canny = cv.Canny(gray, 240, 240)
    contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #Mostrar imagenes con diferentes detecciones de bordes
    cv.imshow("Canny", canny)

    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.drawContours(gray, contours, -1, (255, 0, 0), 3)

    cv.imshow("Contour", gray)

    cv.waitKey(0)
    cv.destroyAllWindows()

    #Loop para lineas de hough
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    cv.imshow("Hough Lines in Building", img)

    cv.waitKey(0)
    cv.destroyAllWindows()

    list_Border = [canny]

    #Implementacion de morfologias
    for j in list_Border:
        # Kernel Definition
        kernel = np.ones((5, 5), np.uint8)
        # Erosion of an Image
        erosion = cv.erode(j, kernel, iterations=1)
        # Dilation of an Image
        dilation = cv.dilate(j, kernel, iterations=1)
        # Opening of an Image (Erosion followed by Dilation)
        opening = cv.morphologyEx(j, cv.MORPH_OPEN, kernel)
        # Closing of an Image (Dilation followed by Erosion)
        closing = cv.morphologyEx(j, cv.MORPH_CLOSE, kernel)
        # Morphological Gradient
        gradient = cv.morphologyEx(j, cv.MORPH_GRADIENT, kernel)
        # Top Hat Image (It is the difference between input image and Opening of the image)
        tophat = cv.morphologyEx(j, cv.MORPH_TOPHAT, kernel)
        # Black Hat (It is the difference between the closing of the input image and input
        blackhat = cv.morphologyEx(j, cv.MORPH_BLACKHAT, kernel)

        list_Morphological = [erosion, dilation, opening, closing, gradient, tophat, blackhat]

        #Mostrar imagenes con morfologias concatenadas
        Morpho = np.concatenate(list_Morphological, axis=1)
        Morpho_Resize = cv.resize(Morpho,(1800,250), interpolation=cv.INTER_AREA)
        cv.imshow("Morphological(erosion, dilation, opening, closing, gradient, tophat, blackhat)", Morpho_Resize)


        cv.waitKey(0)
        cv.destroyAllWindows()


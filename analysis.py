import cv2
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from scipy.ndimage import measurements as ms

channel_width = 113
box = ((1379, 1379+43), (91, 91+53))
x, y = box
background = cv2.imread("background.bmp",cv2.IMREAD_GRAYSCALE)
dataPath = "./data/{}/{}/{}" # ratio/series/oil_flow/
postPath = "./post/{}/{}" # ratio/series/
file_pattern = "{}.tif" #{droplet_number}.tiff


def ImageProcessing(image):
    image = cv2.absdiff(image, background)
    h, gray = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV);
    gray = cv2.medianBlur(gray,5)

    kernel = np.ones((3,3), np.uint8)

    gray = cv2.erode(gray, kernel, iterations=1)

    des = cv2.bitwise_not(gray)
    tmp = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    contour, hier = tmp[1], tmp[0]

    for cnt in contour:
        cv2.drawContours(des,[cnt],0,255,-1)

    gray = cv2.bitwise_not(des)

    gray = cv2.dilate(gray, kernel, iterations=1)

    return gray

if __name__ == "__main__":
    cv2.startWindowThread()
    cv2.namedWindow("preview")

    ratio = 1
    for series in np.arange(1):
        posttmp = postPath.format(ratio, series)
        try:
            os.makedirs(posttmp)
        except:
            pass

        oil_flow = [x.name for x in os.scandir(dataPath[:-2].format(ratio,series))]
        for flow in sorted(oil_flow):
            #sum_flow = float(flow) #+ float(flow)*ratio
            data = glob.glob(dataPath.format(ratio,series,flow)+"/*.tif")
            droplets = []
            for name in data:
                image = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
                image = ImageProcessing(image)

                labeled, nlabels = ms.label(cv2.bitwise_not(image))
                tmp = np.median(labeled[y[0]:y[1], x[0]:x[1]])
                image = 1.0*(labeled == tmp)
                cv2.imshow("preview", image)
                l = np.sum(np.sum(image,axis=0) > 0) / channel_width
                print(l)
                droplets.append(l)

            pickle.dump(droplets, open(posttmp+"/{:.4f}.p".format(float(flow)),"wb"))
    cv2.destroyAllWindows()

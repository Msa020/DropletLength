import cv2
import sys
import signal
import shutil
import time
import os
import numpy as np
import mFsuite.nemesys as nemesys
import mFsuite.miueye as miueye

from time import sleep


# Global Objects
cam, nem = None, None
droplet, oil = None, None

# Parameters
box = ((1379, 1379+43), (91, 91+53))
background = cv2.imread("test.bmp",cv2.IMREAD_GRAYSCALE)
msg_pattern = "{} - ratio: {} - series: {} - oil: {:.4f} ml/h - droplet: {:.4f} ml/h - number: {}"

# Ranges
ratios = [1]
series = 1
num_droplets = 40
flow_range = [0.2, 0.1] #np.arange(0.35, 0.03, -0.01) #np.arange(0.15,0.03, -0.01)

dataPath = "./data/{}_test/{}/{:.4f}" # ratio/series/oil_flow/
file_pattern = "{}.tif" #{droplet_number}.tiff

def Initialization():
    global droplet, oil, cam, nem
    cam = miueye.Camera()
    nem = nemesys.Nemesys()

    nem.Connect()
    nem.RestoreParameters()
    #nem.SyringeParameters((100, 'ul'),(60, 'mm'))
    oil, droplet,_, _ = nem.unit

    cv2.startWindowThread()
    cv2.namedWindow("preview")

    cam.Init()
    cam.ParameterSet('camera.ini')
    cam.Start(25)
    time.sleep(1)

def ExperimentInterrupt(signal, frame, message='Pushed Ctrl-C!'):
    print("*End\t"+message)
    cam.Stop()
    time.sleep(0.2)
    nem.StopAll()
    nem.StoreParameters()
    nem.Disconnect()
    cv2.destroyAllWindows()
    cam.Close()
    sys.exit(0)

def ImageProcessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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

def WaitForDroplet(level, box, max_time = 100*60):
    x, y = box
    waiting = True
    start_time = time.time()
    while waiting:
        raw_image = cam.GetImage()
        image = ImageProcessing(raw_image)
        cv2.imshow("preview", image)
        if time.time()-start_time > max_time:
            ExperimentInterrupt(None, None, "Waiting too long!")
        if np.median(image[y[0]:y[1], x[0]:x[1]]) == level:
            waiting = False
        #else:
        #    time.sleep(0.04)

    return raw_image

def Experiment(dataPath, ratio, series, ndrop, oil_flow, droplet_flow):
    droplet_flow =  oil_flow * ratio
    oil.GenerateConstantFlow((oil_flow, 'ml/h'))
    droplet.GenerateConstantFlow((droplet_flow, 'ml/h'))
    time.sleep(20)
    for n in np.arange(10):
        image = WaitForDroplet(0, box)
        print("Waiting for stabilization: " + str(n))
        WaitForDroplet(255, box)
    
    for n in np.arange(ndrop):
        #Waiting for droplet in specified position
        image = WaitForDroplet(0, box)
        print(msg_pattern.format(time.strftime("%X"), ratio, series, oil_flow, droplet_flow, n))
        cv2.imwrite(dataPath+"/"+file_pattern.format(n), image)
        # Waiting for droplet goes out of position
        WaitForDroplet(255, box)


def ReloadSyringe(syringe, level, no):
    """Reload syringe if level is too small."""
    if syringe.GetActualPosition()/syringe.MaxPosSyringe <= level:
        print("* Reload Syringe - " + no)
        syringe.SwitchValve(1)
        sleep(0.5)
        syringe.GenerateConstantFlow((-20, 'ml/h'))
        sleep(0.5)
        while not syringe.IsDosingFinished():
            sleep(0.5)

        sleep(0.2)
        syringe.SwitchValve(0)
        sleep(0.2)

        return True
    else:
        return False


if __name__ == "__main__":
    signal.signal(signal.SIGINT, ExperimentInterrupt)
    Initialization()

    for r in ratios:
        for s in np.arange(series):
            for flow in flow_range:
                oil_flow = flow
                droplet_flow = flow*r

                datatmp = dataPath.format(r,s,oil_flow)
                try:
                    os.makedirs(datatmp)
                except:
                    pass

       #         ReloadSyringe(oil, 1, "oil")
        #        ReloadSyringe(droplet, 1, "droplet")

                Experiment(datatmp, r, s, num_droplets, oil_flow, droplet_flow)
                time.sleep(0.1)

    nem.StopAll()
    ExperimentInterrupt(None, None,"Experiment Completed!")

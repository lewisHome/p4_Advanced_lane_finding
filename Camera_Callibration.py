## import dependencies
import argparse
import cv2
from glob import glob
import numpy as np


#http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

def calcDistortion(calFP, showImg, nx, ny):
    #termintation criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #load calibration images and find corners
    objectPoints = []
    imgPoints = []

    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    for imageFP in glob(calFP+'/*.jpg'):
        #read image
        img = cv2.imread(imageFP)
        #convert to gray scale
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners=cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objectPoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),criteria)
            imgPoints.append(corners2)
            if showImg==True:
                cv2.drawChessboardCorners(img,(nx, ny), corners2, ret)
                cv2.imshow("Image",img)
                cv2.waitKey(1000)

    #calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imgPoints, gray.shape[::-1], None, None)
    #Calculate callibration Errors
    tot_error=0
    for i in range(len(objectPoints)):
        imgPoints2, _ = cv2.projectPoints(objectPoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgPoints[i],imgPoints2, cv2.NORM_L2)/len(imgPoints2)
        tot_error += error

    meanError = (tot_error/len(objectPoints))
    print ("total error: ", tot_error)
    print ("Mean Error : ", meanError)
    print("Callibration Complete!")
    return mtx, dist, rvecs, tvecs

#Function to store callibration constants
def saveConstants(calFP, mtx, dist, revec, tvec):
    np.save(calFP+'/mtx.npy', mtx, allow_pickle=False)
    np.save(calFP+'/dist.npy', dist, allow_pickle=False)
    np.save(calFP+'/rvecs.npy', rvecs, allow_pickle=False)
    np.save(calFP+'/tvecs.npy', tvecs, allow_pickle=False)
    print('Data Saved!')
    return



if __name__ == '__main__':
    #construct arg parser
    parser = argparse.ArgumentParser(description = 'Camera Callibration')
    parser.add_argument(
        '-camera_images',
        type = str,
        nargs = '?',
        default='Udacity_camera_cal',
        help='Path to folder containing camera callibration images.'
        )
    parser.add_argument(
        '-show_images',
        type = bool,
        nargs = '?',
        default = False,
        help = 'Set this to true if you would like to see corners mapped on images'
        )
    parser.add_argument(
        '-nx',
        type = int,
        nargs = '?',
        default = 9,
        help = 'number of internal corners in x on checkerboard images'
        )
    parser.add_argument(
        '-ny',
        type = int,
        nargs = '?',
        default = 6,
        help = 'number of internal corners in y on checkerboard images'
        )
    args = parser.parse_args()
    
    print(args)
    #Calc camera constants
    mtx, dist, rvecs, tvecs = calcDistortion(args.camera_images, args.show_images, args.nx, args.ny)

    #Show calculated Values
    if args.show_images == True:
        print("MTX: ",mtx)
        print('====================')
        print("DIST: ",dist)
        print('====================')
        print("RVECS: ",rvecs)
        print('====================')
        print("TVECS: ",tvecs)

    #Save Camera constants
    saveConstants(args.camera_images, mtx, dist, rvecs, tvecs)

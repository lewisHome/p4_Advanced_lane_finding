#Ref 1 https://github.com/parilo/carnd-advanced-lane-line-finding/blob/master/Advanced-Lane-Line-Finding-v2.ipynb
#NOTES currently you are trying to figure out how to draw the polynomial on to the perspective image
#Hopefully you have found the equations to fit using find curvature function
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from LFutils import perspectiveFinder, Line
import time

#This function loads camera constants from camera_callibration.py output files
def loadCameraConstants(camera_data):
    try:
        mtx = np.load(camera_data + "\\mtx.npy")
        dist = np.load(camera_data + "\\dist.npy")
        return mtx, dist
    except IOError:
        print("Camera callibration data not avaliable - please run Camera_callibration.py")
        exit

#This function loads any predefined perspective distortion constants
def loadPerspectiveConstants(filePath):
    try:
        filePath = ((filePath.rstrip('.mp4')).rstrip('.jpg'))
        M = np.load(filePath + '_M.npy')
        Minv = np.load(filePath + '_Minv.npy')
        PDims = np.load(filePath + '_PDims.npy')
        print("Perspective constants found for :",filePath)
    except IOError:
        M = None
        Minv = None
        PDims = None
    return M, Minv, PDims

#This function uses camera constants loaded from loadCameraConstants to unwarp image
def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

#First threshhold the frame in the HLS colour space then find edges in threshold image
def threshFrame(image):
    dimy, dimx, dimz = image.shape
    #convert image to HLS colour space
    HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #split HLS image into seperate channels
    H,L,S = cv2.split(HLS)
    Hmask = cv2.threshold(H,75,255,cv2.THRESH_BINARY_INV)[1]
    Lmask = cv2.threshold(L,127,255,cv2.THRESH_BINARY)[1]
    Smask = cv2.threshold(S,150,255,cv2.THRESH_BINARY)[1]
    #find edges in each threshold colour space
    #find edges in each threshold colour space
    Hmask = cv2.Canny(Hmask,100,200)
    Lmask = cv2.Canny(Lmask,100,200)
    Smask = cv2.Canny(Smask,100,200)
    #return masked images
    return Hmask, Lmask, Smask 

#This function performs a perspective transform on an image to get a top down view
def perspectiveTransform(image, M, PDims):
    warp = cv2.warpPerspective( image, \
                                M, \
                                (PDims[0],PDims[1]), \
                                flags=cv2.INTER_LINEAR)
    return  warp

#this function has been ripped from Udacity course notes it draws a sliding window mask onto a blank image
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

#this function has been ripped from Udacity course notes it is used to initialise the lane lines
def find_window_centroids(image):
    window_width=80
    window_height = 80
    margin = 100
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(image)
        r_points = np.zeros_like(image)

        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,image,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,image,window_centroids[level][1],level)
	    # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        outputLeft = cv2.bitwise_and(l_points,image)
        outputRight = cv2.bitwise_and(r_points,image)
    return outputLeft, outputRight
    
       

#This function draws the lane on the origional input image
def drawLane(image,fitx,yList):
    points=np.asarray(np.vstack((fitx, yList)).T,dtype=np.uint16)
    draw=np.zeros_like(image)
    cv2.fillPoly(draw,np.int_([points]), color=255)
    return draw

#This draws the lane lines on the perspective transformed image
def drawLine(image, fit0 ,fit1, fit2):
    yList=np.linspace(0,image.shape[0], num=10, endpoint=True,dtype=np.float32)
    fitx = fit0*yList**2 + fit1*yList + fit2
    fitx = np.asarray(fitx,dtype=np.uint16)
    points=np.asarray(np.vstack((fitx, yList)).T,dtype=np.uint16)
    draw=np.zeros_like(image)
    cv2.polylines(draw,np.int_([points]),False,color=255, thickness = 5)
    return draw

#This function fits a polynomial to the masked lane lines to find the curvature of the lane lines
def findCurvature(image):
    points=np.nonzero(image)
    if len(points[0])>100:
        fit, residual, _,_,_  = np.polyfit(points[0], points[1], 2, full=True)
        conf=1- residual/(points[1].size * points[1].var())
        fit  = np.polyfit(points[0], points[1], 2)
        yList=np.linspace(0,image.shape[0], num=10, endpoint=True,dtype=np.float32)
        fitx = fit[0]*yList**2 + fit[1]*yList + fit[2]
        fitx = np.asarray(fitx,dtype=np.uint16)
#        fitx = np.asarray(fitx,dtype=np.uint16)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(points[0]*ym_per_pix, points[1]*xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*int(points[0].max())*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        #calculate intersection point
        intersect = fit[0]
    else:
        intersect = 0
        curverad = 0
        fit = [0,0,0]
        conf=0
    return curverad,intersect, fit, conf

#This function applys a msak around the region where the lane was found in the preceding image
def quickMask(image, fit, lane):
    yList=np.linspace(0,image.shape[0], num=10, endpoint=True,dtype=np.float32)
    fitx = fit[0]*yList**2 + fit[1]*yList + fit[2]
    if lane =='left':
        fitx=fitx+250
        fitx=np.append(fitx,[0,0])
        yList = np.append(yList,[image.shape[0],0])
    else:
        fitx=fitx-250
        fitx=np.append(fitx,[image.shape[1],image.shape[1]])
        yList = np.append(yList,[image.shape[0],0])
    Mask = drawLane(image,fitx,yList)
    yListl = np.linspace(0,image.shape[0], num=10, endpoint=True,dtype=np.float32)
    fitxl = fit[0]*yListl**2 + fit[1]*yListl + fit[2]-50
    yListr = np.flipud(yListl)
    fitxr = fit[0]*yListr**2 + fit[1]*yListr + fit[2]+50
    yList=np.append(yListl,yListr)
    fitx = np.append(fitxl,fitxr)
    confMaskLine=np.zeros_like(image)
    confMaskLine = drawLane(confMaskLine, fitx, yList)
    mask = cv2.bitwise_and(image,confMaskLine)
    return mask

#This function determines how confident we can be in the lane line found
def confidence(image, fit, lane):
    if fit == None:
        conf=0
    else:
        yList=np.linspace(0,image.shape[0], num=10, endpoint=True,dtype=np.float32)
        fitx = fit[0]*yList**2 + fit[1]*yList + fit[2]
        if lane =='left':
            fitx=fitx+250
            fitx=np.append(fitx,[0,0])
            yList = np.append(yList,[image.shape[0],0])
        else:
            fitx=fitx-250
            fitx=np.append(fitx,[image.shape[1],image.shape[1]])
            yList = np.append(yList,[image.shape[0],0])

        confMask = drawLane(image,fitx,yList)
        yListl = np.linspace(0,image.shape[0], num=10, endpoint=True,dtype=np.float32)
        fitxl = fit[0]*yListl**2 + fit[1]*yListl + fit[2]-50
        yListr = np.flipud(yListl)
        fitxr = fit[0]*yListr**2 + fit[1]*yListr + fit[2]+50
        yList=np.append(yListl,yListr)
        fitx = np.append(fitxl,fitxr)
        confMaskLine=np.zeros_like(image)
        confMaskLine = drawLane(confMaskLine, fitx, yList)
        confMaskLineInv = cv2.bitwise_not(confMaskLine)
        confMask = cv2.bitwise_and(confMask, confMaskLineInv)
        confImgOut=cv2.bitwise_and(image,confMask)
        confImgIn = cv2.bitwise_and(image,confMaskLine)
        conf = np.divide((np.sum(confImgIn)+1),np.sum(confImgOut))
    return conf

#This function finds the lane lines in the image
def findLanes(image):
    if lineData.lineFound == 0:
        leftLane, rightLane =find_window_centroids(image)
    else:
        leftLane = quickMask(image,lineData.leftPoly, 'left')
        rightLane = quickMask(image, lineData.rightPoly, 'right')
    leftCurveRad, leftIntersect, leftFit, leftConf = findCurvature(leftLane)
    leftConf = confidence(image, leftFit, 'left')
    rightCurveRad, rightIntersect, rightFit, rightConf = findCurvature(rightLane)
    rightConf = confidence(image, rightFit, 'right')
    return [leftConf, leftCurveRad, leftFit[0], leftFit[1], leftFit[2], rightConf, rightCurveRad, rightFit[0], rightFit[1], rightFit[2]]


#This function chooses which channel has found the best lane line
def chooseLine(calcLaneParameters):
    '''
        return [leftConf, leftCurveRad, leftFit[0], leftFit[1], leftFit[2], rightConf, rightCurveRad, rightFit[0], rightFit[1], rightFit[2]]
    '''
    leftLineFound = False
    rightLineFound = False
    lcChoose=np.argmax(calcLaneParameters,axis=0)[0]
    rightChannelSort = np.argsort(-calcLaneParameters[:,5],axis=0)
    #The L channel often only shows 1 lane to prevent this function from returning the same lane as left and right
    #this if statement ensures that this doesn't happen
    if rightChannelSort[0] != 1:
        rcChoose=rightChannelSort[0]
    else:
        rcChoose=rightChannelSort[1]
 
    if calcLaneParameters[lcChoose,0] > 5:
        lineData.leftPoly=calcLaneParameters[lcChoose,2:5]
        lineData.leftInt = calcLaneParameters[lcChoose,4]
        lineData.leftCurve=calcLaneParameters[lcChoose,1]
        leftLineFound = True
    else:
        pass

    if calcLaneParameters[rcChoose,5] > 5:
        lineData.rightPoly=calcLaneParameters[rcChoose,7:10]
        lineData.rightInt = calcLaneParameters[rcChoose,9]
        lineData.rightCurve=calcLaneParameters[rcChoose,6]
        rightLineFound = True
    else:
        pass

    if leftLineFound and rightLineFound:
        lineData.lineFound = 1
    elif leftLineFound and not rightLineFound:
        lineData.lineFound = 2
    elif not leftLineFound and rightLineFound:
        lineData.lineFound = 3
    else:
        lineData.lineFound = 4
    return lcChoose, rcChoose

def videoProcess(img):
    img = undistort(img, mtx, dist)
    Hmask,Lmask,Smask=threshFrame(img)
    if M == None:
        imgPerspective = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)

        cv2.putText(imgPerspective, \
                    "Press c to callibrate", \
                    (0,50), \
                    cv2.FONT_HERSHEY_DUPLEX, \
                    2, \
                    (255,255,255))
        
        cv2.putText(imgPerspective, \
                    "perspective transform constants", \
                    (0,100), \
                    cv2.FONT_HERSHEY_DUPLEX, \
                    2, \
                    (255,255,255))

        runAnalysis = False
        
    else:
        HPerspective = perspectiveTransform(Hmask, M, PDims)
        
        LPerspective = perspectiveTransform(Lmask, M, PDims)
        
        SPerspective = perspectiveTransform(Smask, M, PDims)
        
        runAnalysis = True

    if runAnalysis == True:
        '''
            [leftConf, leftCurveRad, leftFit[0], leftFit[1], leftFit[2], rightConf, rightCurveRad, rightFit[0], rightFit[1], rightFit[2]]
        '''
        calcLaneParameters=findLanes(HPerspective)
        calcLaneParameters=np.vstack((calcLaneParameters,findLanes(LPerspective)))
        calcLaneParameters=np.vstack((calcLaneParameters,findLanes(SPerspective)))
        lineData.params = calcLaneParameters
        leftChannel, rightChannel = chooseLine(calcLaneParameters)

        if lineData.lineFound > 0:
            warpMask=np.zeros_like(HPerspective)
            yListl = np.linspace(0,warpMask.shape[0], num=10, endpoint=True,dtype=np.float32)
            fitxl = lineData.leftPoly[0]*yListl**2 + lineData.leftPoly[1]*yListl + lineData.leftPoly[2]
            yListr = np.flipud(yListl)
            fitxr = lineData.rightPoly[0]*yListr**2 + lineData.rightPoly[1]*yListr + lineData.rightPoly[2]
            yList=np.append(yListl,yListr)
            fitx = np.append(fitxl,fitxr)
            warpMask=drawLane(warpMask,fitx,yList)

            Mask = cv2.warpPerspective(warpMask, \
                                       Minv, \
                                      (PDims[0],PDims[1]), \
                                       flags=cv2.INTER_LINEAR)
            Mask = np.lib.pad(Mask,((0,0),(10,10)),'minimum')
            #Draw Lane lines onto output image
            MaskRGB = np.dstack((np.zeros_like(Mask),Mask,np.zeros_like(Mask)))
            img = cv2.addWeighted(img,1, MaskRGB, 0.3, 0)
            radOfCurvature = 'Radius of curvature is {:}m'.format(int(lineData.leftCurve))
            img = cv2.putText(img,radOfCurvature,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,0),2,cv2.LINE_AA)

            centreLine = abs(warpMask.shape[1] - ((lineData.rightInt-lineData.leftInt)/2))*(3.7/7000)

            if centreLine < Mask.shape[1]:
                centreLineText = 'Vehicle is {:.2f}m left of centre'.format(centreLine)
            else:
                centreLineText = 'Vehicle is {:.2f}m right of centre'.format(centreLine)
            img = cv2.putText(img,centreLineText,(50,150), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,0),2,cv2.LINE_AA)

        leftDisp=np.concatenate((HPerspective,LPerspective),axis=0)
        leftDisp=np.concatenate((leftDisp,SPerspective),axis=0)
        leftDisp=cv2.cvtColor(leftDisp,cv2.COLOR_GRAY2BGR)
        rightDisp=leftDisp
        
        dispMask=np.zeros_like(leftDisp)
        maskStart=leftChannel*HPerspective.shape[0]
        maskStop=(leftChannel+1)*HPerspective.shape[0]
        if lineData.lineFound in (1,2):
            dispMask[maskStart:maskStop,:,:]=(0,255,0)
        else:
            dispMask[maskStart:maskStop,:,:]=(0,0,255)
        leftDisp = cv2.addWeighted(leftDisp,1, dispMask, 0.3, 0)

        dispMask=np.zeros_like(rightDisp)
        maskStart=rightChannel*HPerspective.shape[0]
        maskStop=(rightChannel+1)*HPerspective.shape[0]
        if lineData.lineFound in(1,3):
            dispMask[maskStart:maskStop,:,:]=(0,255,0)
        else:
            dispMask[maskStart:maskStop,:,:]=(0,0,255)
        rightDisp = cv2.addWeighted(rightDisp,1, dispMask, 0.3, 0)

        if lineData.lineFound > 0:
            #Draw left lane lines
            leftLinesDisp = np.concatenate((drawLine(HPerspective,lineData.params[0,2],lineData.params[0,3],lineData.params[0,4]),drawLine(LPerspective,lineData.params[1,2],lineData.params[1,3],lineData.params[1,4])),axis=0)
            leftLinesDisp = np.concatenate((leftLinesDisp,drawLine(LPerspective,lineData.params[2,2],lineData.params[2,3],lineData.params[2,4])),axis=0)
            leftLinesDisp = np.dstack((leftLinesDisp,np.zeros_like(leftLinesDisp),np.zeros_like(leftLinesDisp)))
            leftDisp = cv2.addWeighted(leftDisp,1, leftLinesDisp, 1, 0)

            #display left lane confidence parameter
            leftDisp = cv2.putText(leftDisp,'H Conf: = {:.2f}'.format(float(lineData.params[0,0])),(5,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(250,250,250),2,cv2.LINE_AA)
            leftDisp = cv2.putText(leftDisp,'L Conf: = {:.2f}'.format(float(lineData.params[1,0])),(5,770), cv2.FONT_HERSHEY_SIMPLEX, 2,(250,250,250),2,cv2.LINE_AA)
            leftDisp = cv2.putText(leftDisp,'S Conf: = {:.2f}'.format(float(lineData.params[2,0])),(5,1490), cv2.FONT_HERSHEY_SIMPLEX, 2,(250,250,250),2,cv2.LINE_AA)

            #draw right Lanes
            rightLinesDisp = np.concatenate((drawLine(HPerspective,lineData.params[0,7],lineData.params[0,8],lineData.params[0,9]),drawLine(LPerspective,lineData.params[1,7],lineData.params[1,8],lineData.params[1,9])),axis=0)
            rightLinesDisp = np.concatenate((rightLinesDisp,drawLine(LPerspective,lineData.params[2,7],lineData.params[2,8],lineData.params[2,9])),axis=0)
            rightLinesDisp = np.dstack((rightLinesDisp,np.zeros_like(rightLinesDisp),np.zeros_like(rightLinesDisp)))
            rightDisp = cv2.addWeighted(rightDisp,1, rightLinesDisp, 1, 0)
            
            #dispaly right lane confidence parameter
            rightDisp = cv2.putText(rightDisp,'H Conf: = {:.2f}'.format(float(lineData.params[0,6])),(5,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(250,250,250),2,cv2.LINE_AA)
            rightDisp = cv2.putText(rightDisp,'L Conf: = {:.2f}'.format(float(lineData.params[1,6])),(5,770), cv2.FONT_HERSHEY_SIMPLEX, 2,(250,250,250),2,cv2.LINE_AA)
            rightDisp = cv2.putText(rightDisp,'S Conf: = {:.2f}'.format(float(lineData.params[2,6])),(5,1490), cv2.FONT_HERSHEY_SIMPLEX, 2,(250,250,250),2,cv2.LINE_AA)

        leftDisp = cv2.resize(leftDisp,None, fx=0.3333333333, fy=0.3333333333, interpolation = cv2.INTER_CUBIC)
        rightDisp = cv2.resize(rightDisp,None, fx=0.3333333333, fy=0.3333333333, interpolation = cv2.INTER_CUBIC)
        displayImg = np.concatenate((leftDisp,img),axis=1)
        displayImg = np.concatenate((displayImg,rightDisp),axis=1)
        displayImg = cv2.resize(displayImg,None, fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
    else:
        displayImg = np.concatenate((img,imgPerspective),axis=1)
        displayImg = cv2.resize(displayImg,None, fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)                    

    return displayImg

#def imageProjection
if __name__ == '__main__':
    #construct arg parser
    parser = argparse.ArgumentParser(description = 'Lane Finder')
    parser.add_argument(
        '-camera_data',
        type = str,
        nargs = '?',
        default = 'Udacity_camera_cal',
        help = 'Path to file containing camera calibration data'
        )
    parser.add_argument(
        '-test_data',
        type = str,
        help = 'Image or video for processing'
        )
    parser.add_argument(
        '-output_file',
        type = str,
        nargs='?',
        default = 'output_images/',
        help = 'File to store processed Data'
        )

    args = parser.parse_args()
    # load camera callibration constants
    global lineData #This holds the data of the last found line
    global mtx, dist #camera distortion constants
    global M, Minv, PDims #Perspective distortion constants
    global imH, imW #This is a global variable for the video frame width and frame height
    global pwarpH, pwarpW #These are global variables for the warped image width and height
    lineData = Line()

    #load camera and perspective constants
    mtx, dist = loadCameraConstants(args.camera_data)
    M , Minv, PDims = loadPerspectiveConstants(args.test_data)

    # When not recording run images through opencv - setup your perspective transform
    # using the perspective transform finder
    cap = cv2.VideoCapture(args.test_data)
    setUpOutput = False
    while cap.isOpened():
        ret, img = cap.read()
        if ret == False:
            break
        displayImg = videoProcess(img)
        cv2.imshow("Image",displayImg)
        # Define the codec and create VideoWriter object
        if setUpOutput == False:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            videoOut = args.output_file+args.test_data
            print("Saving processed video to: ", videoOut)
            imageName = args.output_file+args.test_data.rstrip('mp4')+'jpg'
            print("Saving initial image to: ", imageName)
            cv2.imwrite(imageName,displayImg)
            setUpOutput = True
            out = cv2.VideoWriter(videoOut,fourcc, 20.0, (displayImg.shape[1],displayImg.shape[0]))
        out.write(displayImg)
        #pause to allow image to render and look for user input
        key = cv2.waitKey(25)
        # press q to quit
        if key == 113:
            break
        # press p to pause video and store image
        if key == 112:
            imageName=time.strftime("%j%H%M%S")+".jpg"
            print(imageName)
            cv2.imwrite(imageName,displayImg)
            cv2.waitKey(0)
        #run c to callibrate perspective transform
        elif key == 99:
            M, Minv, PDims = perspectiveFinder(img,args.test_data)                                
    cap.release()
    out.release()
    cv2.destroyAllWindows()

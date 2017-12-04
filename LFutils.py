import numpy as np
import cv2

class Line():
    def __init__(self):
        self.lineFound = 0
        self.leftPoly = None
        self.rightPoly = None
        self.leftCurve = None
        self.rightCurve = None
        self.params = None
        
def callback(x):
    return

def perspectiveFinder(image,filePath):
    cv2.destroyAllWindows()
    #These are rough values which roughly align perspective transform for camera on udacity vehicle
    imH, imW = image.shape[:2]
    centreLine = int(imW/2)+10
    depthLine = int(24*imH/64)
    upperWidth = int(11*imW/64)
    lowerWidth = imW-20

    leftLim = 0
    rightLim = imW
    upperLim = 0
    lowerLim = imH
    
    cv2.namedWindow("Perspective Finder Input")
      
    cv2.createTrackbar("Centre Line Track Bar",
                       "Perspective Finder Input",
                       centreLine,
                       imW,
                       callback)

    cv2.createTrackbar("Depth Line Track Bar",
                       "Perspective Finder Input",
                       depthLine,
                       imH,
                       callback)

    cv2.createTrackbar("Upper Width Track Bar",
                       "Perspective Finder Input",
                       upperWidth,
                       imW,
                       callback)

    cv2.createTrackbar("Lower Width Track Bar",
                       "Perspective Finder Input",
                       lowerWidth,
                       imW,
                       callback)
    
    cv2.namedWindow("Perspective Finder Output")
      
    cv2.createTrackbar("Left Limit",
                       "Perspective Finder Output",
                       leftLim,
                       imW,
                       callback)

    cv2.createTrackbar("Right Limit",
                       "Perspective Finder Output",
                       rightLim,
                       imW,
                       callback)

    cv2.createTrackbar("Upper Limit",
                       "Perspective Finder Output",
                       upperLim,
                       imH,
                       callback)

    cv2.createTrackbar("Lower Limit",
                       "Perspective Finder Output",
                       lowerLim,
                       imH,
                       callback)

    while True:

        print("IMW before: ", imW)
        print("shape before: ", image.shape)
        print("Lower width before: ", lowerWidth)
        centreLine = cv2.getTrackbarPos("Centre Line Track Bar",
                       "Perspective Finder Input")
        
        depthLine = cv2.getTrackbarPos("Depth Line Track Bar",
                       "Perspective Finder Input")
        
        upperWidth = cv2.getTrackbarPos("Upper Width Track Bar",
                       "Perspective Finder Input")
        
        lowerWidth = cv2.getTrackbarPos("Lower Width Track Bar",
                       "Perspective Finder Input")
        
        leftLim = cv2.getTrackbarPos("Left Limit",
                       "Perspective Finder Output")
        
        rightLim = cv2.getTrackbarPos("Right Limit",
                       "Perspective Finder Output")
        
        upperLim = cv2.getTrackbarPos("Upper Limit",
                       "Perspective Finder Output")
        
        lowerLim = cv2.getTrackbarPos("Lower Limit",
                       "Perspective Finder Output")

        imageDraw = image

        print("IMW after: ", imW)
        print("shape after: ", image.shape)
        print("Lower width after: ", lowerWidth)
        #array of corner points upper left, upper right, lower right, lower left adjustable using
        #trackbars above input image
        pts = np.array([[centreLine-(upperWidth/2),imH-depthLine],
                        [centreLine+(upperWidth/2),imH-depthLine],
                        [centreLine+(lowerWidth/2),imH],
                        [centreLine-(lowerWidth/2),imH]], np.int32)
        
        pts = pts.reshape((-1,1,2))
        
        cv2.polylines(imageDraw,[pts],True,(0,0,255))
        imageDraw=cv2.resize(imageDraw, None,fx =0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        cv2.putText(imageDraw, \
            "Press q to exit", \
            (0,50), \
            cv2.FONT_HERSHEY_DUPLEX, \
            2, \
            (255,255,255))

        cv2.putText(imageDraw, \
            "Press s to save", \
            (0,100), \
            cv2.FONT_HERSHEY_DUPLEX, \
            2, \
            (255,255,255))
 
        cv2.imshow("Perspective Finder Input", imageDraw)

        # array of transofrm points top left, top rightbotom left, topright, bottom right
        ptsIn=pts.astype(np.float32)
        # array of transform points top left, botom left, topright, bottom right adjustable using
        # track bars above output image
        ptsOut = np.array([[leftLim,upperLim],
                        [rightLim, upperLim],
                        [rightLim, lowerLim],
                        [leftLim, lowerLim]], np.float32)
        ptsOut = ptsOut.reshape((-1,1,2))
        print(ptsIn)
        print("================================")
        print(ptsOut)
        M = cv2.getPerspectiveTransform(ptsIn, ptsOut)
        print("================================")
        imageOut = cv2.warpPerspective(image, M, (lowerWidth,imH), flags=cv2.INTER_LINEAR)
        imageOut = cv2.resize(imageOut, None,fx =0.5,fy =0.5, interpolation = cv2.INTER_CUBIC)
        cv2.imshow("Perspective Finder Output", imageOut)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return None, None, None
        elif cv2.waitKey(25) & 0xFF == ord('s'):
            Minv = cv2.getPerspectiveTransform(ptsOut, ptsIn)
            PDims = (lowerWidth,imH)
            np.save((filePath.rstrip('.mp4')+'_M.npy'), M, allow_pickle=False)
            np.save((filePath.rstrip('.mp4')+'_Minv.npy'), Minv, allow_pickle=False)                
            np.save((filePath.rstrip('.mp4')+'_PDims.npy'), PDims, allow_pickle=False)                
            cv2.destroyAllWindows()
            return M, Minv, PDims
    

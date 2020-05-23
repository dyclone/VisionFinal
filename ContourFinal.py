import numpy as np
import cv2

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass
'''
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 150, 255,empty)
cv2.createTrackbar("Threshold2", "Parameters", 255, 255,empty)
cv2.createTrackbar("Area", "Parameters", 0, 30000, empty)
'''
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img, imgContour):
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (0, 0, 255), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            if len(approx) == 3:
                shape = "triangle"
            elif len(approx) == 4:
                ar = w / float(h)
                shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
            elif len(approx) == 5:
                shape = "pentagon"
            elif len(approx) == 6:
                shape = "hexagon"
            elif len(approx) == 7 or len(approx) == 8:
                shape = "circle"
            else:
                shape = "unknown shape"

            cv2.putText(imgContour, shape, (x + int(w/2) - 10, y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (255, 255, 255), 2)

def get_test_params(a_file):
    tester = open(a_file)
    inputs = []
    for x in tester:
        line_read = []
        for word in x.split():
            line_read.append(word)
        inputs.append(line_read[1:])
    tester.close()
    return inputs

def shape_detection(a_file_name):
    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 640, 240)
    cv2.createTrackbar("Threshold1", "Parameters", 150, 255, empty)
    cv2.createTrackbar("Threshold2", "Parameters", 255, 255, empty)
    cv2.createTrackbar("Area", "Parameters", 0, 30000, empty)

    while True:
        # Read Camera
        # success, img = cap.read()
        img = cv2.imread(a_file_name)
        img = cv2.resize(img, (540, 380))

        # Blur and Convert camera feed to gray scale
        imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

        # Use canny edge detection
        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

        # Remove noise and overlap from canny edge detection
        kernel = np.ones((5, 5))
        imgdil = cv2.dilate(imgCanny, kernel, iterations=1)

        # Use imgCanny to develop contours and put the results in the original image
        imgContour = img.copy()
        getContours(imgCanny, imgContour)

        # Use imgDil to develop contours and put the results in the original image
        imgContour2 = img.copy()
        getContours(imgdil, imgContour2)

        # Store the results at each step of getting an image, finding the contours and identifying shapes
        imgStack = stackImages(0.8, ([img, imgGray, imgCanny],
                                     [imgdil, imgContour, imgContour2]))

        # Display the results as long as you dont press q
        cv2.imshow("Result", imgStack)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

def main():
    inputs = get_test_params("testparameters.txt")
    test_num = int(input("Enter the test you would like to run:")) - 1
    test_image = inputs[test_num][0]
    shape_detection(test_image)

if __name__ == '__main__':
    main()


'''
while True:
    #Read Camera
    #success, img = cap.read()
    img = cv2.imread("shapes_and_colors.jpg")
    img = cv2.resize(img, (540, 380))

    #Blur and Convert camera feed to gray scale
    imgBlur = cv2.GaussianBlur(img, (7,7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    #Use canny edge detection
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    #Remove noise and overlap from canny edge detection
    kernel = np.ones((5,5))
    imgdil = cv2.dilate(imgCanny, kernel, iterations=1)

    #Use imgCanny to develop contours and put the results in the original image
    imgContour = img.copy()
    getContours(imgCanny, imgContour)

    #Use imgDil to develop contours and put the results in the original image
    imgContour2 = img.copy()
    getContours(imgdil, imgContour2)

    #Store the results at each step of getting an image, finding the contours and identifying shapes
    imgStack = stackImages(0.8,([img, imgGray, imgCanny],
                                [imgdil, imgContour, imgContour2]))

    #Display the results as long as you dont press q
    cv2.imshow("Result", imgStack)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
'''
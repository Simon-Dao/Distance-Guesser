import cv2
import numpy as np

cv2.namedWindow("main")
cv2.moveWindow("main",200,200)

cv2.namedWindow("HSV")
cv2.moveWindow("HSV",750,200)
cv2.resizeWindow("HSV",500,400)

cv2.namedWindow("hsvResult")

width = 500
height = 500

procHeight = 700
procWidth = 700

known_width = 8
known_distance = 24
known_pixels = 85

path = "C:\\Users\\Gamer\\Desktop\\ring detection test photos\\chair-"

def empty(a):
    pass

#w = width of the object in inches
#f =
def getDist(w,p):
    f = (known_pixels * known_distance) / known_width

    return (w * f)/p

class Rect:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

cv2.createTrackbar("H_MIN", "HSV", 0, 255, empty)
cv2.createTrackbar("H_MAX", "HSV", 108, 255, empty)
cv2.createTrackbar("S_MIN", "HSV", 167, 255, empty)
cv2.createTrackbar("S_MAX", "HSV", 255, 255, empty)
cv2.createTrackbar("V_MIN", "HSV", 109, 255, empty)
cv2.createTrackbar("V_MAX", "HSV", 255, 255, empty)
cv2.createTrackbar("w", "HSV", 570, 2000, empty)
cv2.createTrackbar("pic", "HSV", 1, 7, empty)

while True:
    pic = 1 if cv2.getTrackbarPos('pic','HSV') == 0 else cv2.getTrackbarPos('pic','HSV')
    src = cv2.imread(path+str(pic)+'.jpg')

    src = cv2.resize(src, (cv2.getTrackbarPos("w", "HSV"), cv2.getTrackbarPos("w", "HSV")))

    #convert colors from rgb to hsv
    cv2.GaussianBlur(src,(5,5),0)
    hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)

    h_min = cv2.getTrackbarPos("H_MIN", "HSV")
    h_max = cv2.getTrackbarPos("H_MAX", "HSV")
    s_min = cv2.getTrackbarPos("S_MIN", "HSV")
    s_max = cv2.getTrackbarPos("S_MAX", "HSV")
    v_min = cv2.getTrackbarPos("V_MIN", "HSV")
    v_max = cv2.getTrackbarPos("V_MAX", "HSV")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)
    hsv = cv2.bitwise_and(hsv,hsv, mask = mask)

    #find contours
    contourMat = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)
    contours,_ = cv2.findContours(contourMat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #get largest blob of yellow and draw a rect around it
    largestRect = Rect(0,0,0,0)
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        rect = Rect(x,y,w,h)

        if largestRect.area() < rect.area():
            largestRect = rect

    x = largestRect.x
    y = largestRect.y
    w = largestRect.width
    h = largestRect.height

    cv2.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)
    print(w)

    #find the distance
    dist = (getDist(known_width, w))

    src = cv2.resize(src, (width, height))
    hsv = cv2.resize(hsv, (width,height))

    cv2.putText(src,str(round(dist,2))+' inches',(200,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("main",src)
    cv2.imshow("hsvResult", hsv)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
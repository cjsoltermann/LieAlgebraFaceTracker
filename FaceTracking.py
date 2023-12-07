import cv2
import numpy as np
from scipy.linalg import expm
from sklearn.linear_model import Ridge

def begin_tracking():
    global tracking, warped
    
    face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face[top_left_pt[1]:bottom_right_pt[1],top_left_pt[0]:bottom_right_pt[0]], (r,r))
  
    m = np.zeros([n,3,3])
    
    m[:,0:2,:] = np.random.uniform(-l, l, [n,2,3])
    
    M = expm(m)
    M[:,0:2,2] *= r
    hog_size = hogger.compute(face, (r,r)).shape[0]

    center_zero = np.array([[1,0,-r / 2],[0,1,-r / 2],[0,0,1]])
    uncenter_zero = np.linalg.inv(center_zero)
    M = uncenter_zero @ M @ center_zero

    hogs = np.zeros((n, hog_size))
    for i in range(n):
        warped = cv2.warpAffine(face, M[i,0:2], (r, r), 0, 0, cv2.BORDER_WRAP)
        hogs[i] = hogger.compute(warped, (r,r))

    y = m[:,0:2,:].reshape((-1, 6))
    X = hogs

    model.fit(X, y)
    
    tracking = True

def track():
    global top_left_pt, bottom_right_pt, tracking, cut
    if (tracking):
        for i in range(10):
            face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = face[max(0,top_left_pt[1]):bottom_right_pt[1],max(0,top_left_pt[0]):bottom_right_pt[0]]
            
            if (face.shape[0] == 0 
                or face.shape[1] == 0
                or abs(top_left_pt[0]) > 10000 
                or abs(top_left_pt[1]) > 10000 
                or abs(bottom_right_pt[0]) > 10000 
                or abs(bottom_right_pt[1]) > 10000):
                
                tracking = False
                top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
                return
                
            face = cv2.resize(face, (r,r))
            
            cut = face.copy()
            
            hog = hogger.compute(face, (r,r))
            delta_m = np.zeros((3,3))
            delta_m[0:2] = model.predict(hog[None])[0].reshape((2,3))
            delta_M = expm(delta_m)
            #print(delta_M)

            dx = int(delta_M[0,2] * r)
            dy = int(delta_M[1,2] * r)

            sx = delta_M[0,0]
            sy = delta_M[1,1]

            dw = int((bottom_right_pt[0] - top_left_pt[0]) * sx) - (bottom_right_pt[0] - top_left_pt[0])
            dh = int((bottom_right_pt[1] - top_left_pt[1]) * sy) - (bottom_right_pt[1] - top_left_pt[1])
                        
            top_left_pt = top_left_pt[0] + dx - dw // 2, top_left_pt[1] + dy - dh // 2
            bottom_right_pt = bottom_right_pt[0] + dx + dw // 2, bottom_right_pt[1] + dy + dh // 2
            
            #bottom_right_pt = top_left_pt[0] + w, top_left_pt[1] + h


def handle_mouse(event, x, y, flags, param):
    global drawing, tracking, top_left_pt, bottom_right_pt

    x = frame.shape[1] - x

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            bottom_right_pt = (x,y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        tracking = False
        drawing = True
        top_left_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        
        if (bottom_right_pt[0] < top_left_pt[0]):
            tmp = bottom_right_pt[0]
            bottom_right_pt = (top_left_pt[0], bottom_right_pt[1])
            top_left_pt = (tmp, top_left_pt[1])

        if (bottom_right_pt[1] < top_left_pt[1]):
            tmp = bottom_right_pt[1]
            bottom_right_pt = (bottom_right_pt[0], top_left_pt[1])
            top_left_pt = (top_left_pt[0], tmp)
            
        begin_tracking()

drawing = False
tracking = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
region_mat = np.zeros((3,3))
cut = []

    
l = 0.08 # Range of random Lie vectors
n = 400 # Number of initial samples
r = 100 # Size of training images

model = Ridge()
hogger = cv2.HOGDescriptor()

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

cv2.namedWindow('Image')
cv2.imshow('Image', frame)

cv2.setMouseCallback('Image', handle_mouse)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if tracking:
        track()

    # points = np.array([[-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1]]) @ region_mat.transpose()
    # cv2.polylines(frame, points, False, 2)
    cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)

    cv2.imshow('Image', cv2.flip(frame, 1))

    if len(cut) != 0:
        cv2.imshow('Cut', cv2.flip(cut,1))
    
    # Break the loop when 'ESC' key is pressed
    if cv2.waitKey(1) == 27:
        break
    if cv2.getWindowProperty('Image',cv2.WND_PROP_VISIBLE) < 1:        
        break

cap.release()
cv2.destroyAllWindows()
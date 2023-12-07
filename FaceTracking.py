import cv2
import numpy as np
from scipy.linalg import expm
from sklearn.linear_model import Ridge

l = 0.08 # Range of random Lie vectors
n = 400 # Number of initial samples
r = 100 # Size of training images

class State:
    def __init__(self, frame) -> None:
        self.drawing = False
        self.rect_start = (-1, -1)
        self.rect_end = (-1, -1)

        self.tracking = False
        self.frame_width = frame.shape[1]
        self.frame_height = frame.shape[0]
        self.unit_to_frame_mat = cv2.getAffineTransform(np.array([[-1, -1], [-1, 1], [1, 1]], np.float32), np.array([[0, self.frame_height], [0, 0], [self.frame_width, 0]], np.float32))
        self.unit_to_frame_mat = np.append(self.unit_to_frame_mat, [[0,0,1]], axis=0)
        self.frame_to_unit_mat = np.linalg.inv(self.unit_to_frame_mat)
        self.unit_to_workspace_mat = cv2.getAffineTransform(np.array([[-1, -1], [-1, 1], [1, 1]], np.float32), np.array([[0, r], [0, 0], [r, 0]], np.float32))
        self.unit_to_workspace_mat = np.append(self.unit_to_workspace_mat, [[0,0,1]], axis=0)
        self.workspace_to_unit_mat = np.linalg.inv(self.unit_to_workspace_mat)
        self.unit_to_region_mat = np.zeros((3,3))
        self.cut = None
        self.cur_frame = frame
        self.model = Ridge()
        self.hogger = cv2.HOGDescriptor()

    def region_outside_frame(self):
        test_points_unit = np.array([[-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]])
        test_points_frame = test_points_unit @ self.unit_to_region_mat.transpose() @ self.frame_to_unit_mat.transpose()
        for point in test_points_frame:
            if abs(point[0]) < 1 or abs(point[1]) < 1:
                return False
        return True

def main():

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    state = State(frame)

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cut', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', frame)

    cv2.setMouseCallback('Image', lambda event, x, y, flags, param : handle_mouse(event, x, y, flags, param, state))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        state.cur_frame = frame
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if state.tracking:
            track(state)

        if state.tracking:
            points = np.array([[-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1]]) @ state.unit_to_region_mat.transpose()
            points = points.astype(np.int64)
            cv2.polylines(frame, points[None, :, 0:2], True, (0, 255, 0), 2)

        if state.drawing:
            cv2.rectangle(frame, state.rect_start, state.rect_end, (0, 255, 0), 2)

        cv2.imshow('Image', cv2.flip(frame, 1))

        if not state.cut is None:
            cv2.imshow('Cut', cv2.flip(state.cut,1))
    
        # Break the loop when 'ESC' key is pressed
        if cv2.waitKey(1) == 27:
            break
        if cv2.getWindowProperty('Image',cv2.WND_PROP_VISIBLE) < 1:        
            break

    cap.release()
    cv2.destroyAllWindows()

def make_unit_to_rect(top_left_pt, bottom_right_pt):
    p = np.array([[-1, -1], [-1, 1], [1, 1]], np.float32)
    q = np.array([[top_left_pt[0], bottom_right_pt[1]], [top_left_pt[0], top_left_pt[1]], [bottom_right_pt[0], top_left_pt[1]]], np.float32)

    return np.append(cv2.getAffineTransform(p, q), [[0,0,1]], axis=0)

def begin_tracking(state: State):
    face = cv2.cvtColor(state.cur_frame, cv2.COLOR_BGR2GRAY)
    face = cv2.warpAffine(state.cur_frame, (state.unit_to_region_mat @ state.workspace_to_unit_mat)[0:2], (r, r), 0, cv2.WARP_INVERSE_MAP)
  
    m = np.zeros([n,3,3])
    m[:,0:2,:] = np.random.uniform(-l, l, [n,2,3])
    
    M = expm(m)

    hog_size = state.hogger.compute(face, (r,r)).shape[0]

    hogs = np.zeros((n, hog_size))
    for i in range(n):
        warped = cv2.warpAffine(face, (state.unit_to_workspace_mat @ M[i] @ state.workspace_to_unit_mat)[0:2], (r, r), 0, 0, cv2.BORDER_WRAP)
        hogs[i] = state.hogger.compute(warped, (r,r))

    y = m[:,0:2,:].reshape((-1, 6))
    X = hogs

    state.model.fit(X, y)
    state.tracking = True

def track(state: State):
    if (state.tracking):
        for i in range(10):
            if (state.region_outside_frame()):                
                state.tracking = False
                state.unit_to_region_mat = np.zeros((3,3))
                return

            face = cv2.cvtColor(state.cur_frame, cv2.COLOR_BGR2GRAY)
            face = cv2.warpAffine(state.cur_frame, (state.unit_to_region_mat @ state.workspace_to_unit_mat)[0:2], (r,r), 0, cv2.WARP_INVERSE_MAP, cv2.BORDER_CONSTANT)
            
            state.cut = face.copy()
            
            hog = state.hogger.compute(face, (r,r))
            delta_m = np.zeros((3,3))
            delta_m[0:2] = state.model.predict(hog[None])[0].reshape((2,3))
            delta_M = expm(delta_m)

            state.unit_to_region_mat = state.unit_to_region_mat @ delta_M


def handle_mouse(event, x, y, flags, param, state: State):

    x = state.frame_width - x

    if event == cv2.EVENT_MOUSEMOVE:
        if state.drawing:
            state.rect_end = (x,y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        state.tracking = False
        state.drawing = True
        state.rect_start = (x, y)
        state.rect_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        state.drawing = False
        state.rect_end = (x, y)
        
        if (state.rect_end[0] < state.rect_start[0]):
            tmp = state.rect_end[0]
            state.rect_end = (state.rect_start[0], state.rect_end[1])
            state.rect_start = (tmp, state.rect_start[1])

        if (state.rect_end[1] < state.rect_start[1]):
            tmp = state.rect_end[1]
            state.rect_end = (state.rect_end[0], state.rect_start[1])
            state.rect_start = (state.rect_start[0], tmp)
            
        state.unit_to_region_mat = make_unit_to_rect(state.rect_start, state.rect_end)
        begin_tracking(state)


main()
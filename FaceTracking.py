import cv2
import numpy as np
from scipy.linalg import expm
from sklearn.linear_model import Ridge

# Constants
L = 0.08 # Range of elements of random Lie vectors
N = 400 # Number of initial samples
R = 100 # Size of square training images

## Points defining the unit space
# Used to define affine transformations, which uniquely map triangles
UNIT_TRIANGLE = np.array([[-1, -1], [-1, 1], [1,1]], np.float32)
# Used for checking if the tracking region is completely outside of frame
HOMOGENEOUS_UNIT_SQUARE = np.array([[-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]])

class State:
    def __init__(self, frame) -> None:
        # The image and shape of the current frame of video
        self.cur_frame = frame
        self.frame_width = frame.shape[1]
        self.frame_height = frame.shape[0]

        # The cutout image shown in a separate window
        self.cut = None

        # Whether the user is currently drawing a rectangle and where it is
        self.drawing = False
        self.rect_start = (-1, -1)
        self.rect_end = (-1, -1)

        # Whether an object is currently being tracked
        self.tracking = False

        # A lot of transformation matrices between spaces
        # There are 3 spaces
            # Unit Square with (-1, -1) bottom left corner and (1, 1) top right corner
            # Frame space with (0, 0) top left corner and (frame_width, frame_height) bottom right corner
            # Workspace with (0, 0) top left corner and (r, r) bottom right corner

        # Between the unit sqaure space and the frame space
        self.unit_to_frame_mat = cv2.getAffineTransform(UNIT_TRIANGLE, np.array([[0, self.frame_height], [0, 0], [self.frame_width, 0]], np.float32))
        self.unit_to_frame_mat = np.append(self.unit_to_frame_mat, [[0,0,1]], axis=0)
        self.frame_to_unit_mat = np.linalg.inv(self.unit_to_frame_mat)

        # Between the unit square space and the workspace
        self.unit_to_workspace_mat = cv2.getAffineTransform(UNIT_TRIANGLE, np.array([[0, R], [0, 0], [R, 0]], np.float32))
        self.unit_to_workspace_mat = np.append(self.unit_to_workspace_mat, [[0,0,1]], axis=0)
        self.workspace_to_unit_mat = np.linalg.inv(self.unit_to_workspace_mat)

        # Between the unit square and the tracked region
        self.unit_to_region_mat = np.zeros((3,3))

        # Linear regression model and HOG
        self.model = Ridge()
        self.hogger = cv2.HOGDescriptor()

    def region_outside_frame(self):
        test_points_frame = HOMOGENEOUS_UNIT_SQUARE  @ self.unit_to_region_mat.transpose() @ self.frame_to_unit_mat.transpose()
        # for point in test_points_frame:
        #     if abs(point[0]) < 1 or abs(point[1]) < 1:
        #         return False
        # return True
        return (abs(test_points_frame) > 1).any(1).all()

# Main function
def main():

    # Begin video capture and get first frame
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # Initialize state
    state = State(frame)

    # Create windows
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cut', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', frame)

    # Enable mouse interaction
    cv2.setMouseCallback('Image', lambda event, x, y, flags, param : handle_mouse(event, x, y, flags, param, state))

    # Main loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # If something went wrong
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Place frame into state
        state.cur_frame = frame

        # If a region is currently being tracked
        if state.tracking:
            # Compute the transformation
            track(state)
            # Calculate the edges of the region
            points = HOMOGENEOUS_UNIT_SQUARE @ state.unit_to_region_mat.transpose()
            points = points.astype(np.int64)
            # Draw the region
            cv2.polylines(frame, points[None, :, 0:2], True, (0, 255, 0), 2)

        # If the user is drawing a region
        if state.drawing:
            # Draw the proto-region
            cv2.rectangle(frame, state.rect_start, state.rect_end, (0, 255, 0), 2)

        # Put the frame on the window
        cv2.imshow('Image', cv2.flip(frame, 1))

        # Put the cutout on the cutout window, if a cutout exists
        if not state.cut is None:
            cv2.imshow('Cut', cv2.flip(state.cut,1))
    
        # Break the loop when 'ESC' key is pressed
        if cv2.waitKey(1) == 27:
            break
        if cv2.getWindowProperty('Image',cv2.WND_PROP_VISIBLE) < 1:        
            break

    # Destroy windows and close camera when done
    cap.release()
    cv2.destroyAllWindows()

# Create a transformation that maps the unit square to the rectangle defined by two points
def make_unit_to_rect(top_left_pt, bottom_right_pt):
    # Create triangle out of rect
    region = np.array([[top_left_pt[0], bottom_right_pt[1]], [top_left_pt[0], top_left_pt[1]], [bottom_right_pt[0], top_left_pt[1]]], np.float32)

    # Get the affine transformation
    return np.append(cv2.getAffineTransform(UNIT_TRIANGLE, region), [[0,0,1]], axis=0)

# Initialize tracking: Train model off of N random transformations of the initial region
def begin_tracking(state: State):
    # Convert to grayscale
    face = cv2.cvtColor(state.cur_frame, cv2.COLOR_BGR2GRAY)
    # Place the selected region into the workspace
    face = cv2.warpAffine(state.cur_frame, (state.unit_to_region_mat @ state.workspace_to_unit_mat)[0:2], (R, R), 0, cv2.WARP_INVERSE_MAP)
  
    # Create a random vector in the Lie Algebra
    m = np.zeros([N,3,3])
    m[:,0:2,:] = np.random.uniform(-L, L, [N,2,3])
    
    # Exponentitate the random vector to get a random transformation
    M = expm(m)

    # Calculate the size of the hog vectors
    hog_size = state.hogger.compute(face, (R,R)).shape[0]

    # Compute N random transformations of the workspace
    hogs = np.zeros((N, hog_size))
    for i in range(N):
        warped = cv2.warpAffine(face, (state.unit_to_workspace_mat @ M[i] @ state.workspace_to_unit_mat)[0:2], (R, R), 0, 0, cv2.BORDER_WRAP)
        hogs[i] = state.hogger.compute(warped, (R,R))

    # Get the variables ready for regression
    y = m[:,0:2,:].reshape((-1, 6))
    X = hogs

    # Fit the model and enable tracking
    state.model.fit(X, y)
    state.tracking = True

# Track one frame: Calculate the transformation to apply to the region for the current frame
def track(state: State):
    # Compute 10 iterations to hopefully achieve stability
    for i in range(10):
        # Quit tracking if the region is outside of the frame
        if (state.region_outside_frame()):                
            state.tracking = False
            state.unit_to_region_mat = np.zeros((3,3))
            return

        # Grayscale frame and move face into workspace
        face = cv2.cvtColor(state.cur_frame, cv2.COLOR_BGR2GRAY)
        face = cv2.warpAffine(state.cur_frame, (state.unit_to_region_mat @ state.workspace_to_unit_mat)[0:2], (R,R), 0, cv2.WARP_INVERSE_MAP, cv2.BORDER_CONSTANT)
        
        # Create a copy to show in second window
        state.cut = face.copy()
        
        # Compute the HOG of the workspace
        hog = state.hogger.compute(face, (R,R))
        # Use the model to predict the Lie vector of the transformation given the HOG
        delta_m = np.zeros((3,3))
        delta_m[0:2] = state.model.predict(hog[None])[0].reshape((2,3))
        # Exponentiate the Lie vector to get the transformation
        delta_M = expm(delta_m)

        # Apply the transformation to the tracking region
        state.unit_to_region_mat = state.unit_to_region_mat @ delta_M


# Handle mouse events
def handle_mouse(event, x, y, flags, param, state: State):

    # Image is displayed mirroed, so mirror x location
    x = state.frame_width - x

    # If the mouse is moved and the user is drawing, update the region being drawn
    if event == cv2.EVENT_MOUSEMOVE:
        if state.drawing:
            state.rect_end = (x,y)
    # If the mouse is pressed, start drawing
    elif event == cv2.EVENT_LBUTTONDOWN:
        state.tracking = False
        state.drawing = True
        state.rect_start = (x, y)
        state.rect_end = (x, y)

    # If the mouse is released, finish the drawing and start tracking
    elif event == cv2.EVENT_LBUTTONUP:
        state.drawing = False
        state.rect_end = (x, y)
        
        # Region may be drawn upside down or backwards. Standardize it
        if (state.rect_end[0] < state.rect_start[0]):
            tmp = state.rect_end[0]
            state.rect_end = (state.rect_start[0], state.rect_end[1])
            state.rect_start = (tmp, state.rect_start[1])

        if (state.rect_end[1] < state.rect_start[1]):
            tmp = state.rect_end[1]
            state.rect_end = (state.rect_end[0], state.rect_start[1])
            state.rect_start = (state.rect_start[0], tmp)
            
        # Calculate the initial transformation from the unit square to the tracking region
        state.unit_to_region_mat = make_unit_to_rect(state.rect_start, state.rect_end)
        # Begin tracking
        begin_tracking(state)

# Call main function
main()
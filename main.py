import cv2
import mediapipe as mp
import pyautogui
import random
from util import get_angle, get_distance
from pynput.mouse import Button, Controller
mouse = Controller()

# Get screen dimensions with pyautogui
screen_width, screen_height = pyautogui.size()
print(screen_width, screen_height)

# Cam dimensions
cam_width = 640
cam_height = 480

# For bounding box
frame_red = 100 # Frame reduction

# Initializing model
mpHands = mp.solutions.hands
# Setting up model
hands = mpHands.Hands(
    static_image_mode=False, # False b/c we are capturing a video
    model_complexity=1,  #
    min_detection_confidence=0.7,# 70% confidence required
    min_tracking_confidence=0.7, # 70% confidence required
    max_num_hands=1 # Only one hand being detected
)

# Function for finding the tip of a finger
def find_finger_tip(processed):
    # if processed
    if processed.multi_hand_landmarks:
        # Take the landmarks from one of the hands
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        # Get the index finger tip and store (x,y coord)
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        # Return the index
        return index_finger_tip
    # Return none if no processed landmark action
    return None, None


# Function to move mouse
# Using pyautogui API
def move_mouse(index_finger_tip):
    # If index finger tip exists
    if index_finger_tip is not None:
        # Get x and y values related to screen width
        # Mediapipe knows these screen dimensions
        x = int(index_finger_tip.x * (screen_width))
        y = int(index_finger_tip.y * (screen_height))

        #x = int(index_finger_tip.x * 1920)
        #y = int(index_finger_tip.y * 640)
        # Move our mouse to this XY coordinate of index finger tip
        pyautogui.moveTo(x, y)

# Function that check left click conditions and returns true or false
def is_left_click(landmark_list, thumb_index_dist):
    return (
        # Checks if index finger is bent AND middle finger is straight AND thumb is straight
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 < thumb_index_dist and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90
    )

# Function that check right click conditions and returns true or false
def is_right_click(landmark_list, thumb_index_dist):
    return (
        # Checks if middle finger is bent AND index finger is straight AND thumb is straight
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 < thumb_index_dist and
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90
    )

# Function that check double click conditions and returns true or false
def is_double_click(landmark_list, thumb_index_dist):
    return (
        # Checks if index finger is straight AND middle finger is straight AND thumb is straight
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 < thumb_index_dist and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50
    )

# Function that check screenshot conditions and returns true or false
def is_screenshot(landmark_list, thumb_index_dist):
    return (
        # Checks if all are down (index, thumb, middle)
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist < 50
    )


# Function to detect gestures
# Passing in original frame, the landmark list, and the processed frame
def detect_gesture(frame, landmark_list, processed):
    # If the length of the landmark list is greater than 21 (there is only 21)
    if len(landmark_list) >= 21:
        # 1st action: Move the mouse
        # Need the media pipe to move a point
        # index finger is position of mouse
        # Find the index of finger tip in processed frame
        index_finger_tip = find_finger_tip(processed)
        # Find the distance between landmarks 4 and 5 (thumb closed)
        # 4 and 5 are tips of index and thumb
        thumb_index_dist = get_distance([landmark_list[4], landmark_list[5]])

        # MOVE MOUSE
        # If the thumb and index finger distance is close, and the index finger is upright...
        if get_distance([landmark_list[4], landmark_list[5]]) < 50  and get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            # Move the mouse according to the index finger tip
            move_mouse(index_finger_tip)

        # LEFT CLICK
        elif is_left_click(landmark_list,  thumb_index_dist):
            # Mouse press right
            mouse.press(Button.left)
            # And release
            mouse.release(Button.left)
            # Put text on screen
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # RIGHT CLICK
        elif is_right_click(landmark_list, thumb_index_dist):
            # Mouse press left
            mouse.press(Button.right)
            # And release
            mouse.release(Button.right)
            # Put text on screen
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # DOUBLE CLICK
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif is_screenshot(landmark_list,thumb_index_dist ):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# Main function
def main():
    # Helps draw the landmarks
    draw = mp.solutions.drawing_utils
    # Capture camera 1 with openCV
    cap = cv2.VideoCapture(0)
    cap.set(3, cam_width)
    cap.set(4, cam_height)

    # try catch block so errors are caught
    try:
        # Checking if the capture is running successfully
        while cap.isOpened():
            # Read the video frame by frame
            # If you can read the frame, success bool is true, else false
            success, frame = cap.read()
            # if frame us not read, break out of the loop
            if not success:
                break
            # Else flip the frame (makes it so that it looks like a mirror)
            frame = cv2.flip(frame, 1)

            # Bounding box
            #cv2.rectangle(frame, (frame_red, frame_red), (cam_width - frame_red, cam_height - frame_red), (255, 0, 255), 2)

            # Processing
            # Frame is passed as RGB format as mediapipe requires it (CV captures in BGR)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Pass in frame to model created and stores the processed output
            processed = hands.process(frameRGB)

            # Array to receive all the landmarks from hands
            landmark_list = []
            # If there was detection
            if processed.multi_hand_landmarks:
                # Take the landmarks from one of the hands
                hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
                # Use drawing utility to see the landmarks
                # Pass in actual frame (where), hand landmark (what), and what type we need to draw
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                # Loop through all landmarks detected and push in landmark list
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            # Function to detect gestures (passing in original fame, the landmark list, nand the processed frame!!
            detect_gesture(frame, landmark_list, processed)

            # Show the captured frame
            # Name the window that is showing the frame as "Frame" and pass in frame var
            cv2.imshow('Frame', frame)
            # Wait for 1 ms after each frame is read.
            #  If keyboard q is pressed, break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release the capture and destroy all windows
        cap.release()
        cv2.destroyAllWindows()


# If you import this to any other python file, this function will run
if __name__ == '__main__':
    main()



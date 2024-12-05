import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import random
from util import get_angle, get_distance
from pynput.mouse import Button, Controller
mouse = Controller()
import math
from pynput.keyboard import Controller as KeyboardController, Key
keyboard = KeyboardController()
import time


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


def zoom_slider_control(landmark_list, last_pinch_distance):
    # Calculate current pinch distance between thumb (landmark 4) and index finger (landmark 8)
    pinch_distance = get_distance([landmark_list[4], landmark_list[8]])

    # Threshold for starting zoom (when pointer and thumb are close enough to "touch")
    pinch_threshold = 20

    # Only perform zoom when pinch distance is small enough (fingers are touching)
    if pinch_distance < pinch_threshold:
        # Map pinch distance to zoom level; the further apart, the stronger the zoom in/out
        zoom_level = np.interp(pinch_distance, [0, 100], [-10, 10])  # Adjust zoom sensitivity

        # If the new distance is significantly larger or smaller than the last distance, zoom in or out
        if last_pinch_distance is not None:
            if pinch_distance > last_pinch_distance:
                pyautogui.scroll(int(zoom_level))  # Zoom in (scroll up)
                print("Zooming In")
            elif pinch_distance < last_pinch_distance:
                pyautogui.scroll(int(zoom_level))  # Zoom out (scroll down)
                print("Zooming Out")

        # Update the last pinch distance to track in the next frame
        return pinch_distance
    else:
        # Return None if not pinching
        return None


def is_rotate_gesture(landmark_list):
    # Check angles for all fingers (except the ring finger) being straight (angle > 160 degrees)
    thumb_angle = get_angle(landmark_list[1], landmark_list[2], landmark_list[4])
    index_finger_angle = get_angle(landmark_list[5], landmark_list[6], landmark_list[8])
    middle_finger_angle = get_angle(landmark_list[9], landmark_list[10], landmark_list[12])
    pinky_angle = get_angle(landmark_list[17], landmark_list[18], landmark_list[20])

    # Check if the ring finger is bent (angle < 90 degrees)
    ring_finger_angle = get_angle(landmark_list[13], landmark_list[14], landmark_list[16])

    # If all other fingers are straight, and ring finger is bent, return True for rotate gesture
    return (
                thumb_angle > 160 and index_finger_angle > 160 and middle_finger_angle > 160 and pinky_angle > 160 and ring_finger_angle < 90)


def detect_hand_rotation(landmark_list):
    # Get the angle between thumb and index finger
    thumb_tip = landmark_list[4]  # Thumb tip
    index_tip = landmark_list[8]  # Index finger tip
    thumb_base = landmark_list[2]  # Thumb base for better rotation detection

    # Calculate angle of rotation
    delta_x = index_tip[0] - thumb_base[0]
    delta_y = index_tip[1] - thumb_base[1]
    rotation_angle = math.degrees(math.atan2(delta_y, delta_x))

    # Return the angle
    return rotation_angle


def is_pinky_on_left(landmark_list):
    # Get the coordinates for the pinky tip and index tip
    pinky_tip = landmark_list[mpHands.HandLandmark.PINKY_TIP.value]
    index_tip = landmark_list[mpHands.HandLandmark.INDEX_FINGER_TIP.value]

    # Compare x-coordinates to determine the side of the pinky
    return pinky_tip[0] < index_tip[0]  # Returns True if pinky is on the left


def smooth_rotate_with_keys(rotation_angle, hold_time=0.1):
    if rotation_angle < -30:
        keyboard.press(Key.left)
        time.sleep(hold_time)
        keyboard.release(Key.left)
        print("Rotating Left")
    elif rotation_angle > 30:
        keyboard.press(Key.right)
        time.sleep(hold_time)
        keyboard.release(Key.right)
        print("Rotating Right")
    elif -30 <= rotation_angle <= 30:
        keyboard.press(Key.up)
        time.sleep(hold_time)
        keyboard.release(Key.up)
        print("Rotating Up")


def is_switch_window(landmark_list, threshold=30):
    # Thumb tip = landmark 4, Index finger tip = landmark 8
    thumb_tip = landmark_list[4]
    index_tip = landmark_list[8]

    # Calculate the Euclidean distance between thumb tip and index tip
    distance = get_distance([thumb_tip, index_tip])

    # If the distance is below a certain threshold, return True (they are touching)
    return distance < threshold



last_angle = None  # Initialize for tracking the pointer movement
last_pinch_distance = None


# Function to detect gestures
# Passing in original frame, the landmark list, and the processed frame
def detect_gesture(frame, landmark_list, processed):
    # variables
    global last_pinch_distance
    global last_angle



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

        # SWITCH WINDOW
        elif is_switch_window(landmark_list):
            # Simulate holding down Command and pressing Tab
            pyautogui.keyDown('command')  # Hold down the Command key
            time.sleep(0.2)
            pyautogui.press('tab')  # Press the Tab key
            pyautogui.keyUp('command')  # Release the Command key
            cv2.putText(frame, "Command + Tab", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)



        # ZOOM IN/OUT
        last_pinch_distance = zoom_slider_control(landmark_list, last_pinch_distance)

        # ROTATE
        # Check for rotation gesture (all fingers out except the ring finger)
        if len(landmark_list) >= 21:
            # Check if the gesture for rotation is detected
            if is_rotate_gesture(landmark_list):
                # Get the current hand rotation angle
                rotation_angle = detect_hand_rotation(landmark_list)

                # Determine pinky's position
                if is_pinky_on_left(landmark_list):
                    # Modify rotation logic based on pinky position
                    smooth_rotate_with_keys(-rotation_angle, hold_time=0.1)  # Rotate left if pinky is left
                else:
                    smooth_rotate_with_keys(rotation_angle, hold_time=0.1)  # Rotate right if pinky is right

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
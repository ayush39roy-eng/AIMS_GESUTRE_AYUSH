import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import pyautogui
import numpy as np
import time
import os   
import pickle

# macOS specific: PyAutoGUI is used for mouse movement instead of ctypes
pyautogui.FAILSAFE = False
FRAME_MARGIN = 100
DEAD_ZONE = 3   
ALPHA_SLOW = 0.2
ALPHA_FAST = 0.5
SPEED_THRESHOLD = 15
PINCH_ON = 0.045
PINCH_OFF = 0.07
PINCH_FRAME_THRESHOLD = 4
CLICK_COOLDOWN = 0.3
MOVE_INTERVAL = 0.005 # Adjusted for Mac responsiveness
BLINK_THRESHOLD = 0.6
BLINK_FRAME_THRESHOLD = 2
SCROLL_FRAME_THRESHOLD = 2
SCROLL_SCALE = 5 # Adjusted for Mac scrolling sensitivity
SWIPE_THRESHOLD = 50

# Ensure these files are in the same directory as main.py
HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "face_landmarker.task"

RECORD_FRAMES = 20
GESTURE_THRESHOLD = 0.12
CUSTOM_COOLDOWN = 1.0

# Global states
prev_x, prev_y = 0, 0
last_move_time = 0
pinching = False
pinch_frames = 0
last_click_time = 0
pinch_filtered = 0
blink_frames = 0
scroll_frames = 0
scrolling_active = False
running = True
scroll_y_accum = 0
swipe_frames = 0
swiping_active = False
swipe_cooldown = 0
play_frames = 0
gesture_database = {}
recording = False
record_buffer = []
play_cooldown = 0
last_custom_time = 0
latest_frame = None

screen_w, screen_h = pyautogui.size()

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

if os.path.exists("gestures.pkl"):
    with open("gestures.pkl", "rb") as f:
        gesture_database = pickle.load(f)

def extract_features(landmarks):
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    features = []
    for lm in landmarks:
        features.append(lm.x - base_x)
        features.append(lm.y - base_y)
    return np.array(features)

def main():
    global prev_x, prev_y, last_move_time, pinching, pinch_frames, last_click_time
    global pinch_filtered, blink_frames, scroll_frames, scrolling_active, scroll_y_accum
    global swipe_frames, swiping_active, swipe_cooldown, play_frames, play_cooldown
    global recording, record_buffer, gesture_database, last_custom_time, running, latest_frame

    cap = cv2.VideoCapture(0)
    cap.set(3, 960)
    cap.set(4, 540)

    try:
        while cap.isOpened() and running:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(time.time() * 1000)

            face_result = face_landmarker.detect_for_video(mp_image, timestamp)
            face_detected = len(face_result.face_landmarks) > 0

            if face_detected:
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)

                # Blink Detection for Clicking
                if face_result.face_blendshapes:
                    blendshapes = face_result.face_blendshapes[0]
                    left_blink = next((b.score for b in blendshapes if b.category_name == "eyeBlinkLeft"), 0)
                    right_blink = next((b.score for b in blendshapes if b.category_name == "eyeBlinkRight"), 0)
                    
                    if left_blink > BLINK_THRESHOLD or right_blink > BLINK_THRESHOLD:
                        blink_frames += 1
                    else:
                        blink_frames = 0
                    
                    if blink_frames >= BLINK_FRAME_THRESHOLD and time.time() - last_click_time > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_click_time = time.time()

                if hand_result.hand_landmarks:
                    landmarks = hand_result.hand_landmarks[0]
                    features = extract_features(landmarks)
                    
                    # Recording Custom Gesture Logic
                    if recording:
                        record_buffer.append(features)
                        if len(record_buffer) >= RECORD_FRAMES:
                            template = np.mean(record_buffer, axis=0)
                            name = f"custom_{len(gesture_database)+1}"
                            gesture_database[name] = template
                            with open("gestures.pkl", "wb") as f:
                                pickle.dump(gesture_database, f)
                            recording = False
                            record_buffer.clear()
                    
                    # Custom Gesture Matching
                    if not recording:
                        for name, template in gesture_database.items():
                            distance = np.linalg.norm(features - template)
                            if distance < GESTURE_THRESHOLD and time.time() - last_custom_time > CUSTOM_COOLDOWN:
                                pyautogui.press("volumeup")
                                last_custom_time = time.time()
                                break

                    index_tip = landmarks[8]
                    thumb_tip = landmarks[4]
                    middle_tip = landmarks[12]
                    ring_tip = landmarks[16]
                    pinky_tip = landmarks[20]

                    # Pinch Calculation
                    dx_p, dy_p = index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y
                    pinch_distance = np.sqrt(dx_p**2 + dy_p**2)
                    pinch_filtered += (pinch_distance - pinch_filtered) * 0.3

                    if not pinching and pinch_filtered < PINCH_ON:
                        pinching = True
                    if pinching and pinch_filtered > PINCH_OFF:
                        if pinch_frames > PINCH_FRAME_THRESHOLD and time.time() - last_click_time > CLICK_COOLDOWN:
                            pyautogui.click()
                            last_click_time = time.time()
                        pinching = False
                        pinch_frames = 0
                    if pinching: pinch_frames += 1

                    # Cursor Movement Calculations
                    ix = np.clip(int(index_tip.x * w), FRAME_MARGIN, w - FRAME_MARGIN)
                    iy = np.clip(int(index_tip.y * h), FRAME_MARGIN, h - FRAME_MARGIN)
                    screen_x = np.interp(ix, [FRAME_MARGIN, w-FRAME_MARGIN], [0, screen_w])
                    screen_y = np.interp(iy, [FRAME_MARGIN, h-FRAME_MARGIN], [0, screen_h])
                    
                    dx, dy = screen_x - prev_x, screen_y - prev_y
                    speed = np.sqrt(dx**2 + dy**2)
                    alpha = ALPHA_SLOW if speed < SPEED_THRESHOLD else ALPHA_FAST
                    curr_x = prev_x + (screen_x - prev_x) * alpha
                    curr_y = prev_y + (screen_y - prev_y) * alpha

                    # Gesture States
                    idx_up = index_tip.y < landmarks[6].y
                    mid_up = middle_tip.y < landmarks[10].y
                    ring_up = ring_tip.y < landmarks[14].y
                    pinky_up = pinky_tip.y < landmarks[18].y
                    
                    palm_len = np.hypot(landmarks[0].x - landmarks[9].x, landmarks[0].y - landmarks[9].y)
                    idx_ratio = np.hypot(landmarks[0].x - index_tip.x, landmarks[0].y - index_tip.y) / palm_len
                    mid_ratio = np.hypot(landmarks[0].x - middle_tip.x, landmarks[0].y - middle_tip.y) / palm_len
                    avg_ratio = (idx_ratio + mid_ratio) / 2

                    if not pinching:
                        # Play/Pause Detection (4 fingers up)
                        if idx_up and mid_up and ring_up and pinky_up:
                            play_frames += 1
                            if play_frames > SCROLL_FRAME_THRESHOLD * 2 and time.time() - play_cooldown > 2.0:
                                pyautogui.press('playpause')
                                play_cooldown, play_frames = time.time(), 0
                        
                        # Swipe/App Switch Detection (3 fingers up) - Mac uses Cmd+Tab
                        elif idx_up and mid_up and ring_up and not pinky_up:
                            swipe_frames += 1
                            if swipe_frames > SCROLL_FRAME_THRESHOLD and time.time() - swipe_cooldown > 1.0:
                                dx_swipe = curr_x - prev_x
                                if abs(dx_swipe) > SWIPE_THRESHOLD:
                                    if dx_swipe > 0: pyautogui.hotkey('command', 'tab')
                                    else: pyautogui.hotkey('command', 'shift', 'tab')
                                    swipe_cooldown = time.time()
                        
                        # Cursor Movement (Index up)
                        elif idx_ratio > 1.5 and mid_ratio < 1.2:
                            if abs(curr_x - prev_x) > DEAD_ZONE or abs(curr_y - prev_y) > DEAD_ZONE:
                                if time.time() - last_move_time > MOVE_INTERVAL:
                                    pyautogui.moveTo(curr_x, curr_y) # macOS Native
                                    prev_x, prev_y = curr_x, curr_y
                                    last_move_time = time.time()
                        
                        # Scrolling
                        elif not ring_up and not pinky_up:
                            scroll_frames += 1
                            if scroll_frames > SCROLL_FRAME_THRESHOLD:
                                scroll_val = 1 if avg_ratio > 1.7 else (-1 if avg_ratio < 1.4 else 0)
                                if scroll_val != 0:
                                    pyautogui.scroll(scroll_val * SCROLL_SCALE)
                    else:
                        # Drag while pinching
                        if time.time() - last_move_time > MOVE_INTERVAL:
                            pyautogui.dragTo(curr_x, curr_y, button='left')
                            prev_x, prev_y = curr_x, curr_y
                            last_move_time = time.time()

            if recording:
                cv2.putText(frame, "RECORDING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            latest_frame = frame.copy()
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            if cv2.waitKey(1) & 0xFF == ord('r'):
                recording = True
                record_buffer.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
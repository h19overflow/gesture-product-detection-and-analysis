import mediapipe as mp
import numpy as np
from collections import deque
from collections import Counter

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils  # For drawing landmarks if needed
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
gesture_state = {}
# Gesture smoothing buffer
GESTURE_HISTORY_SIZE = 3
gesture_history = [deque(maxlen=GESTURE_HISTORY_SIZE) for _ in range(2)]  # Separate buffer for two hands

# Define landmark indices for each finger
FINGER_LANDMARKS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20]
}

def get_landmark_coords(hand_landmarks, index):
    """Returns (x, y, z) coordinates of a specific landmark."""
    landmark = hand_landmarks.landmark[index]
    return np.array([landmark.x, landmark.y, landmark.z])

def vector(a, b):
    """Constructs a 3D vector going from point a to point b."""
    return b - a

def angle_between_points(a, b, c):
    """Calculates the angle formed at point b by (a->b) and (b->c)."""
    ab = vector(b, a)
    cb = vector(b, c)
    dot_prod = np.dot(ab, cb)
    mag_ab = np.linalg.norm(ab)
    mag_cb = np.linalg.norm(cb)
    
    if mag_ab * mag_cb == 0:
        return 0.0  # Avoid division by zero
    
    cosine_angle = np.clip(dot_prod / (mag_ab * mag_cb), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def is_finger_extended(hand_landmarks, finger_name, extension_threshold=160, curl_threshold=100):
    """
    Determines if a finger is extended or curled based on joint angles.
    Returns: (extended, curled)
    """
    indices = FINGER_LANDMARKS[finger_name]
    mcp_idx, pip_idx, dip_idx, tip_idx = indices
    mcp = get_landmark_coords(hand_landmarks, mcp_idx)
    pip = get_landmark_coords(hand_landmarks, pip_idx)
    tip = get_landmark_coords(hand_landmarks, tip_idx)

    finger_angle = angle_between_points(mcp, pip, tip)

    extended = finger_angle > extension_threshold
    curled = finger_angle < curl_threshold  # New condition: Finger is curled

    return extended, curled

def palm_orientation(hand_landmarks):
    """Determines the hand orientation (side or top view)."""
    wrist = get_landmark_coords(hand_landmarks, 0)
    middle_base = get_landmark_coords(hand_landmarks, 9)
    palm_vector = vector(wrist, middle_base)
    return np.sign(palm_vector[1])  # Positive = palm up, Negative = palm down


def smooth_gesture(hand_id, gesture, confidence ):
    """
    Smooth gestures using an exponential moving average (EMA) that
    weighs recent detections and their confidence.
    """
    alpha = .3
    if hand_id not in gesture_state:
        gesture_state[hand_id] = {}
    
    # Decay previous gesture confidences.
    for g in gesture_state[hand_id]:
        gesture_state[hand_id][g] *= (1 - alpha)
    
    # Update the current gesture's score.
    gesture_state[hand_id][gesture] = gesture_state[hand_id].get(gesture, 0) + alpha * confidence
    
    # Choose the gesture with the highest smoothed score.
    return max(gesture_state[hand_id], key=gesture_state[hand_id].get)
def detect_gesture(results):
    """
    Classifies gestures for up to 2 hands independently, optimized for speed.
    Returns a dictionary with hand index as the key and the detected gesture as the value.
    """
    hand_gestures = {}

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):  # Process max 2 hands
            thumb_ext, thumb_curled = is_finger_extended(hand_landmarks, "thumb")
            index_ext, index_curled = is_finger_extended(hand_landmarks, "index")
            middle_ext, middle_curled = is_finger_extended(hand_landmarks, "middle")
            ring_ext, ring_curled = is_finger_extended(hand_landmarks, "ring")
            pinky_ext, pinky_curled = is_finger_extended(hand_landmarks, "pinky")

            extended_fingers = sum([thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext])
            curled_fingers = sum([thumb_curled, index_curled, middle_curled, ring_curled, pinky_curled])
            
            # Determine palm orientation
            orientation = palm_orientation(hand_landmarks)
            
            # Gesture classification rules
            if index_ext and not (thumb_ext or middle_ext or ring_ext or pinky_ext):
                detected_gesture = "Pointing"
            elif extended_fingers == 0 or curled_fingers >= 3:  # New condition: If 3+ fingers are curled, count as holding
                detected_gesture = "Holding"
            else:
                detected_gesture = "Neutral"

            # Apply optimized smoothing per hand
            hand_gestures[i] = smooth_gesture(i, detected_gesture,confidence=.3)

    return hand_gestures  # Dictionary of gestures for each detected hand
# MediaPipe Hands: Landmark Index Reference
# -----------------------------------------
# Index | Landmark                   | Description
# -----------------------------------------
#  0    | WRIST                      | Base of the hand (wrist joint)
#  1    | THUMB_CMC                  | Thumb carpometacarpal (base of thumb)
#  2    | THUMB_MCP                  | Thumb metacarpophalangeal (first thumb joint)
#  3    | THUMB_IP                   | Thumb interphalangeal (middle thumb joint)
#  4    | THUMB_TIP                  | Tip of the thumb
#  5    | INDEX_FINGER_MCP           | Index finger metacarpophalangeal (base knuckle)
#  6    | INDEX_FINGER_PIP           | Index finger proximal interphalangeal (middle joint)
#  7    | INDEX_FINGER_DIP           | Index finger distal interphalangeal (near tip)
#  8    | INDEX_FINGER_TIP           | Tip of the index finger
#  9    | MIDDLE_FINGER_MCP          | Middle finger metacarpophalangeal (base knuckle)
# 10    | MIDDLE_FINGER_PIP          | Middle finger proximal interphalangeal (middle joint)
# 11    | MIDDLE_FINGER_DIP          | Middle finger distal interphalangeal (near tip)
# 12    | MIDDLE_FINGER_TIP          | Tip of the middle finger
# 13    | RING_FINGER_MCP            | Ring finger metacarpophalangeal (base knuckle)
# 14    | RING_FINGER_PIP            | Ring finger proximal interphalangeal (middle joint)
# 15    | RING_FINGER_DIP            | Ring finger distal interphalangeal (near tip)
# 16    | RING_FINGER_TIP            | Tip of the ring finger
# 17    | PINKY_MCP                  | Pinky finger metacarpophalangeal (base knuckle)
# 18    | PINKY_PIP                  | Pinky finger proximal interphalangeal (middle joint)
# 19    | PINKY_DIP                  | Pinky finger distal interphalangeal (near tip)
# 20    | PINKY_TIP                  | Tip of the pinky finger

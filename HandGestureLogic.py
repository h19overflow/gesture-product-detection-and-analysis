import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils  # For drawing landmarks if needed
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

def is_finger_extended(hand_landmarks, finger_name, extension_threshold=160, curl_threshold=90):
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

def smooth_gesture(hand_id, gesture):
    """
    Manually smooths gestures using a moving average filter.
    Each hand has its own smoothing buffer.
    """
    global gesture_history
    if hand_id < len(gesture_history):  # Ensure index is within range
        gesture_history[hand_id].append(gesture)

        # Fast NumPy-based mode calculation
        gestures_array = np.array(gesture_history[hand_id])
        unique, counts = np.unique(gestures_array, return_counts=True)
        return unique[np.argmax(counts)]  # Return most frequent gesture
    return gesture  # Fallback

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
            hand_gestures[i] = smooth_gesture(i, detected_gesture)

    return hand_gestures  # Dictionary of gestures for each detected hand

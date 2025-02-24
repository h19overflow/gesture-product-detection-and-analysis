import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import database_queries
from database_queries import update_product_interaction
import os
from PIL import Image
import torch
from collections import defaultdict
from HandGestureLogic import hands, detect_gesture, mp_draw, mp_hands  # Import Hand Gesture Logic
import importlib

# Force reload the module to get new updates
importlib.reload(database_queries)
# Load YOLOv8 model with GPU acceleration if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt").to(device)  # Move YOLO model to GPU if available

# Streamlit UI
st.title("🛍️ Gesture-Based Product Recognition")
st.markdown("📸 **Point or hold an item** to recognize it and display details.")

# UI placeholders
frame_placeholder = st.empty()
product_info_placeholder = st.empty()
product_image_placeholder = st.empty()

# Streamlit button to start/stop live stream
run_stream = st.checkbox("📹 Start Live Stream")

# Hand gesture tracking for multi-hand support
hand_gesture_memory = defaultdict(lambda: "Neutral")  # Tracks last seen gestures per hand
def process_frame(frame):
    """Processes each frame: detects multiple hands, classifies gestures, runs YOLO only when necessary."""
    
    # Downscale the frame for faster processing
    frame = cv2.resize(frame, (640, 480))  
    frame = cv2.flip(frame, 1)  # Mirror the frame for user-friendly display
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hand detection
    results = hands.process(rgb_frame)
    detected_gestures = {}

    detected_product = None
    product_image = None

    # Multi-hand gesture detection with proper tracking
    if results.multi_hand_landmarks:
        detected_gestures = detect_gesture(results)  # Returns a dictionary {0: "Pointing", 1: "Holding", etc.}

        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):  # Limit to 2 hands
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2))

            # Track each hand separately
            hand_gesture_memory[hand_id] = detected_gestures.get(hand_id, "Neutral")  # Default to "Neutral"

    # Determine the final gesture output
    active_gestures = set(hand_gesture_memory.values())

    if "Pointing" in active_gestures:
        most_common_gesture = "Pointing"
    elif "Holding" in active_gestures:
        most_common_gesture = "Holding"
    else:
        most_common_gesture = "Neutral"

    # Display detected gesture on frame
    cv2.putText(frame, f"Gesture: {most_common_gesture}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Run YOLO only when a valid gesture is detected
    if most_common_gesture in ["Pointing", "Holding"]:
        results = model.predict(frame, conf=0.7, device=device)  # Run YOLO on GPU if available
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id, conf = int(box.cls[0]), float(box.conf[0])
                detected_label = model.names[cls_id]

                # Fetch only the required product details
                product_info = database_queries.get_product_details(detected_label)

                if product_info:
                    # Ensure we only extract exactly 7 fields
                    product_data = product_info[:7]  # Slice to prevent errors if more columns exist
                    product_id, name, category, genre, price, description, image_path = product_data

                    detected_product = {
                        "ID": product_id,
                        "Name": name,
                        "Category": category,
                        "Genre": genre,
                        "Price": f"${price}",
                        "Description": description
                    }

                    # 🆕 **Update product interaction in the database**
                    interaction_type = "held" if most_common_gesture == "Holding" else "pointed"
                    update_product_interaction(name, interaction_type)

                    # Load product image
                    if os.path.exists(image_path):
                        product_image = Image.open(image_path)

    return frame, detected_product, product_image


# === RUN LIVE STREAM IN STREAMLIT === #
if run_stream:
    cap = cv2.VideoCapture(0)  # Open webcam

    while run_stream:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠ Webcam not detected!")
            break

        processed_frame, product_info, product_image = process_frame(frame)

        # Display the video frame
        frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)

               # Show product details in a structured UI
        if product_info:
            # 🆕 Fetch interaction stats
            product_analytics = database_queries.get_product_analytics()
            analytics_dict = {p[0]: (p[2], p[3]) for p in product_analytics}  # {Product Name: (Times Held, Times Pointed)}

            times_held, times_pointed = analytics_dict.get(product_info['Name'], (0, 0))

            product_info_placeholder.markdown(
                f"""
                ## 🏷️ **{product_info['Name']}**
                - **🛒 Category:** {product_info['Category']}
                - **🎭 Genre:** {product_info['Genre']}
                - **💲 Price:** {product_info['Price']}
                - **📝 Description:** {product_info['Description']}
                - **🤚 Times Held:** {times_held}
                - **👉 Times Pointed At:** {times_pointed}
                """,
                unsafe_allow_html=True
            )

        # Display the product image separately if available
        if product_image:
            product_image_placeholder.image(product_image, caption=f"🖼️ {product_info['Name']}", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()

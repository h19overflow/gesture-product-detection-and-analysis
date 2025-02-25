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
from HandGestureLogic import hands, detect_gesture, mp_draw, mp_hands  
import importlib

importlib.reload(database_queries)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt").to(device)  

st.title("🛍️ Gesture-Based Product Recognition")
st.markdown("📸 **Point or hold an item** to recognize it and display details.")

frame_placeholder = st.empty()
run_stream = st.checkbox("📹 Start Live Stream")

hand_gesture_memory = defaultdict(lambda: "Neutral")

# UI placeholder for product display
product_display_placeholder = st.empty()  

# ✅ Use session state to persist the last detected product across reruns
if "previous_product" not in st.session_state:
    st.session_state.previous_product = None  

def process_frame(frame):
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    detected_gestures = {}
    detected_product = None
    product_image = None

    if results.multi_hand_landmarks:
        detected_gestures = detect_gesture(results)  

        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2))
            hand_gesture_memory[hand_id] = detected_gestures.get(hand_id, "Neutral")  

    active_gestures = set(hand_gesture_memory.values())
    most_common_gesture = "Pointing" if "Pointing" in active_gestures else "Holding" if "Holding" in active_gestures else "Neutral"

    cv2.putText(frame, f"Gesture: {most_common_gesture}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if most_common_gesture in ["Pointing", "Holding"]:
        results = model.predict(frame, conf=0.7, device=device)  
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id, conf = int(box.cls[0]), float(box.conf[0])
                detected_label = model.names[cls_id]

                product_info = database_queries.get_product_details(detected_label)

                if product_info:
                    product_data = product_info[:7]  
                    product_id, name, category, genre, price, description, image_path = product_data

                    detected_product = {
                        "ID": product_id,
                        "Name": name,
                        "Category": category,
                        "Genre": genre,
                        "Price": f"${price}",
                        "Description": description
                    }

                    interaction_type = "held" if most_common_gesture == "Holding" else "pointed"
                    update_product_interaction(name, interaction_type)

                    if os.path.exists(image_path):
                        product_image = Image.open(image_path)

    return frame, detected_product, product_image


if run_stream:
    cap = cv2.VideoCapture(0)  

    while run_stream:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠ Webcam not detected!")
            break

        processed_frame, product_info, product_image = process_frame(frame)

        frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)

        # ✅ Use session state instead of global variables
        if product_info and product_info != st.session_state.previous_product:
            product_analytics = database_queries.get_product_analytics()
            analytics_dict = {p[0]: (p[2], p[3]) for p in product_analytics}

            times_held, times_pointed = analytics_dict.get(product_info['Name'], (0, 0))

            with product_display_placeholder:
                col1, col2 = st.columns([1, 2])  

                with col1:
                    if product_image:
                        st.image(product_image, caption=f"🖼️ {product_info['Name']}", use_container_width=True)

                with col2:
                    st.markdown(
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

            st.session_state.previous_product = product_info  # ✅ Store product in session state

        elif not product_info and st.session_state.previous_product is not None:
            product_display_placeholder.empty()  # Hide product if no longer detected
            st.session_state.previous_product = None  # Reset tracking

    cap.release()
    cv2.destroyAllWindows()

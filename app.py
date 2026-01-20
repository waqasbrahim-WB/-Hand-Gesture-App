import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

# Page configuration
st.set_page_config(
    page_title="Hand Gesture Cartoon App",
    page_icon="👋",
    layout="wide"
)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #FF6B6B;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .finger-count {
        text-align: center;
        font-size: 4em;
        color: #4ECDC4;
        font-weight: bold;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-title">👋 Hand Gesture Cartoon App 🎨</p>', unsafe_allow_html=True)

# Function to count fingers
def count_fingers(hand_landmarks, handedness):
    """
    Count the number of extended fingers based on hand landmarks
    """
    # Finger tip and PIP landmark indices
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    finger_pips = [6, 10, 14, 18]
    
    fingers_up = []
    
    # Check if right or left hand
    is_right_hand = handedness == "Right"
    
    # Thumb (special case - check horizontal distance)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    
    if is_right_hand:
        if thumb_tip.x < thumb_ip.x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
    else:
        if thumb_tip.x > thumb_ip.x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
    
    # Other four fingers - check if tip is above PIP joint
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
    
    return sum(fingers_up)

# Function to create cartoon character image
def create_cartoon(finger_count):
    """
    Generate a cartoon character based on finger count
    Uses emojis and creative designs
    """
    # Create image
    img = Image.new('RGB', (400, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Cartoon designs for each finger count
    cartoons = {
        0: "✊",
        1: "☝️",
        2: "✌️",
        3: "🤟",
        4: "🖖",
        5: "✋"
    }
    
    # Character descriptions
    characters = {
        0: ("Mr. Fist", "#FF6B6B", "The Strong One"),
        1: ("Pointer Pete", "#4ECDC4", "The Leader"),
        2: ("Peace Patty", "#95E1D3", "The Peaceful"),
        3: ("Rock Randy", "#F38181", "The Cool One"),
        4: ("Spock Sally", "#AA96DA", "Live Long!"),
        5: ("High-Five Harry", "#FCBAD3", "The Friendly")
    }
    
    emoji = cartoons.get(finger_count, "👋")
    name, color, desc = characters.get(finger_count, ("Unknown", "#000000", "Mystery"))
    
    # Draw background circle
    draw.ellipse([50, 50, 350, 350], fill=color, outline="#333333", width=5)
    
    # Draw emoji (simulated with text)
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 120)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw emoji
    draw.text((150, 120), emoji, fill="white", font=font_large)
    
    # Draw name
    draw.text((200, 280), name, fill="white", font=font_medium, anchor="mm")
    
    # Draw description
    draw.text((200, 320), desc, fill="white", font=font_small, anchor="mm")
    
    return img

# Initialize session state
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

# Control buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🎥 Start Camera" if not st.session_state.run_camera else "⏸️ Stop Camera", 
                 use_container_width=True):
        st.session_state.run_camera = not st.session_state.run_camera

# Main application
if st.session_state.run_camera:
    # Create columns for layout
    left_col, right_col = st.columns(2)
    
    # Placeholders
    with left_col:
        st.markdown("### 📹 Camera Feed")
        camera_placeholder = st.empty()
    
    with right_col:
        st.markdown("### 🎨 Cartoon Character")
        finger_count_placeholder = st.empty()
        cartoon_placeholder = st.empty()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        
        current_finger_count = 0
        
        while st.session_state.run_camera:
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Failed to access camera. Please check camera permissions.")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = hands.process(rgb_frame)
            
            finger_count = 0
            
            # Draw hand landmarks and count fingers
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                      results.multi_handedness):
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        rgb_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Count fingers
                    hand_label = handedness.classification[0].label
                    finger_count = count_fingers(hand_landmarks, hand_label)
            
            # Update finger count if changed
            if finger_count != current_finger_count:
                current_finger_count = finger_count
            
            # Add finger count text to frame
            cv2.putText(rgb_frame, f"Fingers: {current_finger_count}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            
            # Display camera feed
            camera_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
            # Display finger count
            finger_count_placeholder.markdown(
                f'<p class="finger-count">{current_finger_count} Fingers</p>',
                unsafe_allow_html=True
            )
            
            # Display cartoon character
            cartoon_img = create_cartoon(current_finger_count)
            cartoon_placeholder.image(cartoon_img, use_container_width=True)
            
            # Small delay to reduce CPU usage
            time.sleep(0.03)
    
    # Release camera
    cap.release()
else:
    # Instructions when camera is off
    st.info("👆 Click 'Start Camera' to begin hand gesture recognition!")
    
    st.markdown("""
    ### 📖 How to Use:
    1. Click the **Start Camera** button above
    2. Show your hand to the camera (1-5 fingers)
    3. Watch the cartoon character change based on finger count!
    
    ### 🎯 Finger Count Guide:
    - **0 Fingers (Fist)** → Mr. Fist - The Strong One ✊
    - **1 Finger** → Pointer Pete - The Leader ☝️
    - **2 Fingers** → Peace Patty - The Peaceful ✌️
    - **3 Fingers** → Rock Randy - The Cool One 🤟
    - **4 Fingers** → Spock Sally - Live Long! 🖖
    - **5 Fingers** → High-Five Harry - The Friendly ✋
    
    ### 💡 Tips:
    - Make sure your hand is clearly visible
    - Keep good lighting for best results
    - Try different hand positions!
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit & MediaPipe</p>",
    unsafe_allow_html=True
)

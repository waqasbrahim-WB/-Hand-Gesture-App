import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Hand Gesture Cartoon App",
    page_icon="👋",
    layout="wide"
)

# Initialize MediaPipe
@st.cache_resource
def load_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    return mp_hands, mp_drawing

mp_hands, mp_drawing = load_mediapipe()

# Custom CSS
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
        font-size: 5em;
        color: #4ECDC4;
        font-weight: bold;
        margin: 20px 0;
    }
    .character-name {
        text-align: center;
        font-size: 2em;
        color: #FF6B6B;
        font-weight: bold;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">👋 Hand Gesture Cartoon App 🎨</p>', unsafe_allow_html=True)

# Function to count fingers
def count_fingers(hand_landmarks, handedness):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    fingers_up = []
    
    is_right_hand = handedness == "Right"
    
    # Thumb
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    
    if is_right_hand:
        fingers_up.append(1 if thumb_tip.x < thumb_ip.x else 0)
    else:
        fingers_up.append(1 if thumb_tip.x > thumb_ip.x else 0)
    
    # Other fingers
    for tip, pip in zip(finger_tips, finger_pips):
        fingers_up.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y else 0)
    
    return sum(fingers_up)

# Function to create cartoon
def create_cartoon(finger_count):
    img = Image.new('RGB', (500, 500), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    
    # Character data
    characters = {
        0: {"emoji": "✊", "name": "Mr. Fist", "color": "#FF6B6B", "desc": "The Strong One"},
        1: {"emoji": "☝️", "name": "Pointer Pete", "color": "#4ECDC4", "desc": "The Leader"},
        2: {"emoji": "✌️", "name": "Peace Patty", "color": "#95E1D3", "desc": "The Peaceful"},
        3: {"emoji": "🤟", "name": "Rock Randy", "color": "#F38181", "desc": "The Cool One"},
        4: {"emoji": "🖖", "name": "Spock Sally", "color": "#AA96DA", "desc": "Live Long!"},
        5: {"emoji": "✋", "name": "High-Five Harry", "color": "#FCBAD3", "desc": "The Friendly"}
    }
    
    char = characters.get(finger_count, characters[0])
    
    # Background circle
    draw.ellipse([50, 50, 450, 450], fill=char["color"], outline="#333333", width=8)
    
    # Inner circle for depth
    draw.ellipse([80, 80, 420, 420], fill=char["color"], outline="white", width=3)
    
    # Draw emoji (large text)
    try:
        # Try to load system fonts
        try:
            emoji_font = ImageFont.truetype("Arial.ttf", 150)
            name_font = ImageFont.truetype("Arial.ttf", 40)
            desc_font = ImageFont.truetype("Arial.ttf", 25)
        except:
            try:
                emoji_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 150)
                name_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
                desc_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
            except:
                emoji_font = ImageFont.load_default()
                name_font = ImageFont.load_default()
                desc_font = ImageFont.load_default()
    except:
        emoji_font = ImageFont.load_default()
        name_font = ImageFont.load_default()
        desc_font = ImageFont.load_default()
    
    # Draw emoji centered
    emoji_bbox = draw.textbbox((0, 0), char["emoji"], font=emoji_font)
    emoji_width = emoji_bbox[2] - emoji_bbox[0]
    emoji_x = (500 - emoji_width) // 2
    draw.text((emoji_x, 130), char["emoji"], fill="white", font=emoji_font)
    
    # Draw name
    name_bbox = draw.textbbox((0, 0), char["name"], font=name_font)
    name_width = name_bbox[2] - name_bbox[0]
    name_x = (500 - name_width) // 2
    draw.text((name_x, 330), char["name"], fill="white", font=name_font)
    
    # Draw description
    desc_bbox = draw.textbbox((0, 0), char["desc"], font=desc_font)
    desc_width = desc_bbox[2] - desc_bbox[0]
    desc_x = (500 - desc_width) // 2
    draw.text((desc_x, 380), char["desc"], fill="white", font=desc_font)
    
    return img, char["name"], char["emoji"]

# Main App
st.markdown("### 📸 Show your hand (1-5 fingers) to the camera!")

# Use Streamlit's camera input
img_file_buffer = st.camera_input("Camera Feed", label_visibility="collapsed")

if img_file_buffer is not None:
    # Read image
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Process with MediaPipe
    rgb_frame = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(rgb_frame)
        
        finger_count = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    rgb_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)
                )
                
                hand_label = handedness.classification[0].label
                finger_count = count_fingers(hand_landmarks, hand_label)
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📹 Camera with Hand Detection")
            # Add finger count overlay
            cv2.putText(rgb_frame, f"Fingers: {finger_count}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
            st.image(rgb_frame, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎨 Your Cartoon Character")
            cartoon_img, char_name, emoji = create_cartoon(finger_count)
            
            st.markdown(f'<p class="finger-count">{finger_count} {emoji}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="character-name">{char_name}</p>', unsafe_allow_html=True)
            st.image(cartoon_img, use_container_width=True)
            
            if finger_count == 0 and results.multi_hand_landmarks:
                st.info("👊 Fist detected! Try opening your hand!")
            elif not results.multi_hand_landmarks:
                st.warning("🤚 No hand detected! Please show your hand clearly to the camera.")

else:
    # Instructions
    st.info("👆 Click the camera button above to take a picture and detect hand gestures!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ✊ 0-1 Fingers")
        st.image(create_cartoon(0)[0], use_container_width=True)
        st.image(create_cartoon(1)[0], use_container_width=True)
    
    with col2:
        st.markdown("### ✌️ 2-3 Fingers")
        st.image(create_cartoon(2)[0], use_container_width=True)
        st.image(create_cartoon(3)[0], use_container_width=True)
    
    with col3:
        st.markdown("### 🖖 4-5 Fingers")
        st.image(create_cartoon(4)[0], use_container_width=True)
        st.image(create_cartoon(5)[0], use_container_width=True)
    
    st.markdown("""
    ### 📖 How to Use:
    1. Click the **camera button** above
    2. Allow camera permissions in your browser
    3. Show your hand (1-5 fingers) to the camera
    4. Click "Take Photo" to capture
    5. See your cartoon character appear!
    
    ### 💡 Tips:
    - Use good lighting for best results
    - Show your full hand clearly
    - Try different finger counts!
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit & MediaPipe | No video processing = Faster deployment!</p>", unsafe_allow_html=True)

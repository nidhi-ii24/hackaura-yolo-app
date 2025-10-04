import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="HackAura - YOLO Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for HackAura theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Header styling */
    header {
        background: rgba(30, 41, 59, 0.95) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #e2e8f0;
        font-size: 1.25rem;
        margin-bottom: 2rem;
    }
    
    /* Badge container */
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .badge {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid #6366f1;
        padding: 0.5rem 1.25rem;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 600;
        color: #a5b4fc;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] h1 {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 800;
    }
            
    /* ALL Headers - Make them bright white */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }
    
    /* Section headers specifically */
    .stMarkdown h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 1rem !important;
    }
    
    /* Stat card styling */
    .stat-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s;
    }
    
    .stat-card:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
    }
    
    /* Feature cards */
    .feature-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.75rem;
    }
    
    .feature-text {
        color: #e2e8f0;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.5);
    }
            
    /* Checkbox styling */
    .stCheckbox label {
        color: #e2e8f0 !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Status indicator */
    .status-active {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid #10b981;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        color: #34d399;
        font-weight: 600;
    }
    
    .status-dot {
        width: 10px;
        height: 10px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    /* Video frame styling */
    .video-frame {
        border: 3px solid #6366f1;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Info box */
    .info-box {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid #6366f1;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #cbd5e1;
    }
            
    /* Sidebar text improvements */
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
            
    /* Slider label */
    .stSlider label {
        color: #e2e8f0 !important;
    }
    
    /* All paragraph text */
    p {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('final.pt')

model = load_model()

# Header
st.markdown('<h1 class="main-title">🔥 HackAura Real-Time Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by YOLOv8 & Computer Vision</p>', unsafe_allow_html=True)

# Badges
st.markdown("""
<div class="badge-container">
    <span class="badge">AI-Powered</span>
    <span class="badge">Live Detection</span>
    <span class="badge">High Accuracy</span>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙ Settings")
    st.markdown("---")
    
    # Confidence threshold
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    st.markdown("---")
    
    # Model info
    st.markdown("### 📊 Model Info")
    st.info("""
    *Model:* YOLOv8 Custom  
    *Type:* Object Detection  
    *Framework:* Ultralytics
    """)
    
    st.markdown("---")
    
    # Instructions
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Click *Start Webcam* below
    2. Allow camera access
    3. Watch real-time detection
    4. Adjust confidence threshold
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📹 Live Detection Feed")
    
    # Status indicator
    st.markdown("""
    <div class="status-active">
        <span class="status-dot"></span>
        <span>Ready to Stream</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Webcam control
    run_webcam = st.checkbox('🎥 Start Webcam', value=False)
    
    # Video placeholder
    FRAME_WINDOW = st.image([])
    
    # Info box
    st.markdown("""
    <div class="info-box">
        💡 <strong>Tip:</strong> Make sure your webcam is connected and browser permissions are granted.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📈 Detection Stats")
    
    # Metrics placeholders
    metric1 = st.empty()
    metric2 = st.empty()
    metric3 = st.empty()
    metric4 = st.empty()

# Run webcam detection
if run_webcam:
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        st.error("❌ Could not access webcam. Please check your camera connection and permissions.")
    else:
        frame_count = 0
        object_count = 0
        
        while run_webcam:
            ret, frame = camera.read()
            if not ret:
                st.warning("⚠ Failed to grab frame from webcam")
                break
            
            # Run YOLO detection
            results = model(frame, conf=confidence)
            
            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Update metrics
            frame_count += 1
            object_count = len(results[0].boxes)
            
            # Display frame
            FRAME_WINDOW.image(frame_rgb, use_container_width=True)
            
            # Update stats in sidebar
            with col2:
                metric1.metric("🎯 Objects Detected", f"{object_count}")
                metric2.metric("📊 Confidence", f"{int(confidence * 100)}%")
                metric3.metric("⚡ Frames Processed", f"{frame_count}")
                metric4.metric("🔍 Status", "Active", delta="Running")
            
            # Small delay to prevent overwhelming
            time.sleep(0.01)
        
        camera.release()

# Features section
st.markdown("---")
st.markdown("## ✨ Key Features")

feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🚀</div>
        <div class="feature-title">Real-Time Processing</div>
        <div class="feature-text">Instant object detection with minimal latency</div>
    </div>
    """, unsafe_allow_html=True)

with feat_col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎓</div>
        <div class="feature-title">Custom Trained Model</div>
        <div class="feature-text">Specialized YOLO model for accurate detection</div>
    </div>
    """, unsafe_allow_html=True)

with feat_col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">💡</div>
        <div class="feature-title">Smart Analytics</div>
        <div class="feature-text">Track and analyze detection patterns</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Built for HackAura Hackathon 2025 | Powered by YOLOv8 & Streamlit
</div>
""", unsafe_allow_html=True)
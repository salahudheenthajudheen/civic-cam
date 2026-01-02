#!/usr/bin/env python3
"""
CivicCam Streamlit Dashboard

Modern, real-time dashboard for the CivicCam AI-powered litter detection system.
Connects to the FastAPI backend for live video feed and incident data.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests
from datetime import datetime
import time

# ==============================================================================
# Configuration
# ==============================================================================

API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 5  # seconds

# Page configuration
st.set_page_config(
    page_title="CivicCam - AI Litter Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Custom CSS Styling
# ==============================================================================

st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background-color: #0a0a0f;
    }
    
    /* Header styling */
    .main-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .logo-text {
        font-size: 20px;
        font-weight: 600;
        color: white;
    }
    
    .status-badge {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 500;
    }
    
    .status-online {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }
    
    .status-offline {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }
    
    /* Stats styling */
    .stats-container {
        display: flex;
        gap: 24px;
        font-size: 13px;
        color: rgba(255, 255, 255, 0.6);
    }
    
    .stat-value {
        color: #00d4ff;
        font-weight: 600;
    }
    
    .stat-fps {
        color: #22c55e;
        font-weight: 600;
    }
    
    /* Live badge */
    .live-badge {
        position: absolute;
        top: 12px;
        left: 12px;
        background: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
        z-index: 10;
    }
    
    .live-dot {
        width: 6px;
        height: 6px;
        background: white;
        border-radius: 50%;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Incident card styling */
    .incident-card {
        display: flex;
        gap: 12px;
        padding: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        cursor: pointer;
        transition: background 0.2s;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    
    .incident-card:hover {
        background: rgba(255, 255, 255, 0.05);
    }
    
    .incident-card.selected {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    .incident-thumbnail {
        width: 48px;
        height: 48px;
        border-radius: 6px;
        object-fit: cover;
        background: #1a1a2e;
    }
    
    .incident-info {
        flex: 1;
        min-width: 0;
    }
    
    .incident-time {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 4px;
    }
    
    .incident-type {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.4);
    }
    
    .incident-plate {
        font-size: 10px;
        color: #00d4ff;
        margin-top: 2px;
    }
    
    /* Detail view styling */
    .detail-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    .detail-title {
        font-size: 14px;
        font-weight: 500;
        color: white;
    }
    
    .panel-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 16px;
    }
    
    .panel-label {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }
    
    .license-plate {
        background: #fef3c7;
        color: #1a1a1a;
        padding: 10px 16px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 2px;
        text-align: center;
        border: 2px solid #1a1a1a;
    }
    
    .not-detected {
        padding: 10px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
        color: rgba(255, 255, 255, 0.3);
        font-size: 11px;
        text-align: center;
    }
    
    .info-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        padding: 16px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 12px;
    }
    
    .info-label {
        color: rgba(255, 255, 255, 0.5);
        margin-bottom: 4px;
    }
    
    .info-value {
        color: white;
        font-weight: 500;
    }
    
    .info-value-success {
        color: #22c55e;
        font-weight: 500;
    }
    
    .info-value-error {
        color: #ef4444;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        padding: 12px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 13px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 16px;
    }
    
    /* Camera offline message */
    .camera-offline {
        height: 400px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: #0a0a0f;
        color: rgba(255, 255, 255, 0.4);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .offline-icon {
        font-size: 48px;
        opacity: 0.3;
        margin-bottom: 16px;
    }
    
    .offline-text {
        font-size: 14px;
    }
    
    .offline-subtext {
        font-size: 12px;
        margin-top: 4px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Video container */
    .video-container {
        position: relative;
        background: #000;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# API Functions
# ==============================================================================

@st.cache_data(ttl=2)
def check_api_health():
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.ok
    except:
        return False


@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_events(limit: int = 20):
    """Fetch recent littering events from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/events?limit={limit}", timeout=5)
        if response.ok:
            return response.json()
    except Exception as e:
        st.error(f"Failed to fetch events: {e}")
    return []


@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_stats():
    """Fetch detection statistics from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=5)
        if response.ok:
            return response.json()
    except Exception as e:
        pass
    return {"totalIncidents": 0, "detectionFps": 0, "activeCameras": 0}


def get_image_url(image_url: str) -> str:
    """Convert relative image URL to absolute."""
    if image_url.startswith("http"):
        return image_url
    return f"{API_BASE_URL}{image_url}"


def send_telegram_notification(event_id: str) -> dict:
    """Send a Telegram notification for an incident."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/send-telegram/{event_id}",
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==============================================================================
# UI Components
# ==============================================================================

def render_header(is_connected: bool, stats: dict):
    """Render the main header with logo, status, and stats."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        status_class = "status-online" if is_connected else "status-offline"
        status_text = "● Live" if is_connected else "○ Offline"
        
        st.markdown(f"""
        <div class="logo-section">
            <span style="font-size: 24px;">🎯</span>
            <span class="logo-text">CivicCam</span>
            <span class="status-badge {status_class}">{status_text}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_incidents = stats.get("totalIncidents", 0)
        fps = stats.get("detectionFps", "—")
        if isinstance(fps, (int, float)):
            fps = f"{fps:.1f}"
        
        st.markdown(f"""
        <div class="stats-container" style="justify-content: flex-end;">
            <span>Incidents: <span class="stat-value">{total_incidents}</span></span>
            <span>FPS: <span class="stat-fps">{fps}</span></span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 16px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)


def render_live_feed(is_connected: bool):
    """Render the live video feed."""
    if not is_connected:
        st.markdown("""
        <div class="camera-offline">
            <span class="offline-icon">📹</span>
            <p class="offline-text">Camera offline</p>
            <p class="offline-subtext">Start API server to view feed</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display live badge and video
    st.markdown("""
    <div class="video-container">
        <div class="live-badge">
            <span class="live-dot"></span>
            LIVE
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # MJPEG stream
    st.image(f"{API_BASE_URL}/api/video", width="stretch")


def render_incident_list(incidents: list, selected_id: str = None):
    """Render the incidents sidebar list."""
    st.markdown('<div class="sidebar-header">📋 Recent Events</div>', unsafe_allow_html=True)
    
    if not incidents:
        st.markdown("""
        <div style="padding: 40px 20px; text-align: center; color: rgba(255,255,255,0.3); font-size: 13px;">
            No incidents yet
        </div>
        """, unsafe_allow_html=True)
        return None
    
    selected = None
    
    for incident in incidents:
        incident_id = incident.get("id", "")
        timestamp = incident.get("timestamp", "Unknown time")
        object_type = incident.get("objectType", "Littering detected")
        plate_number = incident.get("plateNumber")
        image_url = get_image_url(incident.get("imageUrl", ""))
        
        is_selected = incident_id == selected_id
        card_class = "incident-card selected" if is_selected else "incident-card"
        
        # Create a clickable container
        col_img, col_info = st.columns([1, 3])
        
        with col_img:
            try:
                st.image(image_url, width=48)
            except:
                st.markdown("📷")
        
        with col_info:
            st.markdown(f"**{timestamp}**")
            st.caption(object_type)
            if plate_number:
                st.markdown(f"<span style='color: #00d4ff; font-size: 10px;'>🚗 {plate_number}</span>", unsafe_allow_html=True)
        
        if st.button("View Details", key=f"btn_{incident_id}", width="stretch"):
            selected = incident
        
        st.divider()
    
    return selected


def render_incident_detail(incident: dict):
    """Render the detailed incident view."""
    # Header with back button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<span class="detail-title">📋 Incident Details</span>', unsafe_allow_html=True)
    
    with col2:
        if st.button("← Back to Live", width="stretch"):
            st.session_state.selected_incident = None
            st.rerun()
    
    st.divider()
    
    # Main content
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        # Evidence image
        st.markdown('<div class="panel-label">📸 Evidence</div>', unsafe_allow_html=True)
        image_url = get_image_url(incident.get("imageUrl", ""))
        try:
            st.image(image_url, width="stretch")
        except Exception as e:
            st.error("Failed to load evidence image")
    
    with col_side:
        # Suspect face panel
        st.markdown("""
        <div class="panel-card">
            <div class="panel-label">👤 Suspect Face</div>
        </div>
        """, unsafe_allow_html=True)
        
        face_detected = incident.get("faceDetected", False)
        if face_detected:
            try:
                st.image(image_url, width="stretch")
            except:
                st.markdown('<div class="not-detected">Failed to load</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="not-detected">Not detected</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # License plate panel
        st.markdown("""
        <div class="panel-card">
            <div class="panel-label">🚗 License Plate</div>
        </div>
        """, unsafe_allow_html=True)
        
        plate_number = incident.get("plateNumber")
        if plate_number:
            st.markdown(f'<div class="license-plate">{plate_number}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="not-detected">Not detected</div>', unsafe_allow_html=True)
    
    # Metadata grid
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Time**")
        st.write(incident.get("timestamp", "Unknown"))
    
    with col2:
        st.markdown("**Object**")
        st.write(incident.get("objectType", "Unknown"))
    
    with col3:
        st.markdown("**Vehicle**")
        vehicle_detected = incident.get("vehicleDetected", False)
        if vehicle_detected:
            st.markdown('<span style="color: #22c55e;">Yes</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #ef4444;">No</span>', unsafe_allow_html=True)
    
    with col4:
        st.markdown("**Confidence**")
        confidence = incident.get("confidence", 0)
        st.write(f"{int(confidence * 100)}%")
    
    # Send to Telegram button
    st.divider()
    
    col_tg1, col_tg2, col_tg3 = st.columns([1, 2, 1])
    with col_tg2:
        if st.button("📱 Send to Telegram", key="send_telegram_btn", type="primary", width="stretch"):
            incident_id = incident.get("id", "")
            with st.spinner("Sending notification..."):
                result = send_telegram_notification(incident_id)
                if result.get("status") == "sent":
                    st.success("✅ Notification sent to Telegram!")
                elif result.get("status") == "skipped":
                    st.warning("⚠️ " + result.get("message", "Notification skipped"))
                else:
                    st.error("❌ " + result.get("message", "Failed to send"))


# ==============================================================================
# Main Application
# ==============================================================================

def main():
    """Main application entry point."""
    
    # Initialize session state
    if "selected_incident" not in st.session_state:
        st.session_state.selected_incident = None
    
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Check API connection
    is_connected = check_api_health()
    
    # Fetch data
    stats = fetch_stats()
    incidents = fetch_events(20)
    
    # Render header
    render_header(is_connected, stats)
    
    # Main layout
    col_main, col_sidebar = st.columns([3, 1])
    
    with col_sidebar:
        # Render incidents list in sidebar
        for incident in incidents:
            incident_id = incident.get("id", "")
            timestamp = incident.get("timestamp", "Unknown time")
            object_type = incident.get("objectType", "Littering detected")
            plate_number = incident.get("plateNumber")
            image_url = get_image_url(incident.get("imageUrl", ""))
            
            with st.container():
                col_img, col_info = st.columns([1, 2])
                
                with col_img:
                    try:
                        st.image(image_url, width=48)
                    except:
                        st.write("📷")
                
                with col_info:
                    st.markdown(f"<small>{timestamp}</small>", unsafe_allow_html=True)
                    st.caption(object_type)
                
                if st.button("View", key=f"view_{incident_id}", width="stretch"):
                    st.session_state.selected_incident = incident
                    st.rerun()
            
            st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
        
        if not incidents:
            st.info("No incidents detected yet")
    
    with col_main:
        # Main content area
        if st.session_state.selected_incident:
            render_incident_detail(st.session_state.selected_incident)
        else:
            render_live_feed(is_connected)
    
    # Auto-refresh
    time_since_refresh = time.time() - st.session_state.last_refresh
    if time_since_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()

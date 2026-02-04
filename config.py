"""
Configuration file for PCA Face Analysis and Video Call System
Centralizes all configurable parameters for easy management
"""

# ========================== PCA Configuration ==========================
# Number of top eigenfaces to use for compression
TOP_K_EIGENFACES = 1000

# Face dimensions for PCA processing (width, height)
FACE_SIZE = (120, 120)

# File paths for PCA data
EIGENFACES_PATH = "./eigen_faces.npy"
MEAN_FACES_PATH = "./mean_faces.npy"

# ===================== YOLO Model Configuration =====================
# Path to YOLOv8 face detection model
YOLO_MODEL_PATH = "./yolov8n-face-lindevs.pt"

# ==================== Display Configuration ====================
# Display window dimensions (width, height)
DISPLAY_SIZE = (800, 500)

# Face box size for visualization
BOX_SIZE = 200

# Border thickness around face boxes
BORDER_THICKNESS = 10

# Frame color (BGR format)
FRAME_COLOR = (255, 0, 0)  # Blue

# Background color (BGR format)
BACKGROUND_COLOR = (30, 30, 30)  # Dark gray

# ==================== Network Configuration ====================
# Server configuration (for server.py)
# ⚠️ IMPORTANT: You MUST update these IP addresses before running!
# To find your IP address:
#   - Windows: Run 'ipconfig' in command prompt
#   - Linux/Mac: Run 'ifconfig' or 'ip addr' in terminal
# Both computers must be on the same network for P2P communication.

SERVER_IP = "0.0.0.0"  # ⚠️ CHANGE THIS to your local IP address (e.g., 192.168.1.100)
SERVER_PORT = 5142

# Friend's IP for peer-to-peer communication
FRIEND_IP = "127.0.0.1"  # ⚠️ CHANGE THIS to your friend's IP address (e.g., 192.168.1.101)

# Network buffer size (in bytes)
BUFFER_SIZE = 65536

# User names for display
YOUR_NAME = "User1"
FRIEND_NAME = "User2"

# ==================== Camera Configuration ====================
# Camera device index (usually 0 for default webcam)
CAMERA_INDEX = 0

# ==================== Performance Configuration ====================
# Wait key delay in milliseconds (for cv2.waitKey)
WAIT_KEY_DELAY = 1

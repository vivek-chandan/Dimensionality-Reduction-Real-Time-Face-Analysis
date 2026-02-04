"""
PCA-based Video Call System

A peer-to-peer video call system that uses PCA compression to transmit
face data over UDP. Features real-time face detection, compression, and
reconstruction for bandwidth-efficient video communication.
"""

import cv2
import numpy as np
import socket
import json
import threading
import sys

# Import configuration and utilities
import config
from utils import (
    load_pca_data,
    load_yolo_model,
    initialize_camera,
    validate_face_region,
    validate_coordinates,
    compress_face_pca,
    reconstruct_face_pca
)

# ========================== Load Models and Data ==========================
try:
    # Load YOLOv8 face detection model
    model = load_yolo_model(config.YOLO_MODEL_PATH)
    
    # Load precomputed eigenfaces and mean face
    eigenfaces, mean_face = load_pca_data(
        config.EIGENFACES_PATH,
        config.MEAN_FACES_PATH,
        top_k=config.TOP_K_EIGENFACES
    )
    
except Exception as e:
    print(f"Failed to initialize: {e}", file=sys.stderr)
    sys.exit(1)

# ========================== Network Configuration ==========================
# Validate configuration before starting
if config.SERVER_IP == "0.0.0.0":
    print("\n⚠️  ERROR: You must configure your IP address in config.py before running!", file=sys.stderr)
    print("Please update SERVER_IP with your actual local IP address.", file=sys.stderr)
    print("To find your IP:", file=sys.stderr)
    print("  - Windows: Run 'ipconfig' in command prompt", file=sys.stderr)
    print("  - Linux/Mac: Run 'ifconfig' or 'ip addr' in terminal\n", file=sys.stderr)
    sys.exit(1)

if config.FRIEND_IP == "127.0.0.1":
    print("\n⚠️  WARNING: FRIEND_IP is set to localhost (127.0.0.1)", file=sys.stderr)
    print("This will only work if both instances are on the same machine.", file=sys.stderr)
    print("For peer-to-peer communication, set FRIEND_IP to your friend's actual IP address.\n", file=sys.stderr)

# Initialize socket with error handling
try:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((config.SERVER_IP, config.SERVER_PORT))
    # Use 100ms timeout for better balance between responsiveness and CPU efficiency
    server_socket.settimeout(0.1)
    print(f"✓ Server listening on {config.SERVER_IP}:{config.SERVER_PORT}")
except OSError as e:
    print(f"Failed to bind socket: {e}", file=sys.stderr)
    print(f"Make sure the IP address {config.SERVER_IP} is correct and port {config.SERVER_PORT} is available")
    sys.exit(1)

# ========================== Initialize Camera ==========================
try:
    cap = initialize_camera(config.CAMERA_INDEX)
except RuntimeError as e:
    print(f"Camera initialization failed: {e}", file=sys.stderr)
    server_socket.close()
    sys.exit(1)

# ========================== Shared Data with Thread Safety ==========================
# Lock for thread-safe access to shared variables
data_lock = threading.Lock()

# Shared variables for received data
received_compressed_face = None
received_addr = None


def serialize_data(data):
    """
    Serialize numpy array to JSON format (safer than pickle).
    
    Args:
        data (numpy.ndarray): Data to serialize
        
    Returns:
        bytes: Serialized data
    """
    return json.dumps(data.tolist()).encode('utf-8')


def deserialize_data(data_bytes):
    """
    Deserialize JSON data back to numpy array.
    
    Args:
        data_bytes (bytes): Serialized data
        
    Returns:
        numpy.ndarray: Deserialized array
    """
    return np.array(json.loads(data_bytes.decode('utf-8')), dtype=np.float32)


def receive_data():
    """
    Receiver thread function that continuously listens for incoming face data.
    Uses thread-safe access to shared variables.
    """
    global received_compressed_face, received_addr
    
    while True:
        try:
            data, addr = server_socket.recvfrom(config.BUFFER_SIZE)
            
            # Deserialize received data
            compressed_face = deserialize_data(data)
            
            # Update shared variables with thread safety
            with data_lock:
                received_compressed_face = compressed_face
                received_addr = addr
                
        except socket.timeout:
            # Timeout is normal, just continue
            continue
        except json.JSONDecodeError as e:
            print(f"Warning: Received invalid data: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Warning: Error receiving data: {e}", file=sys.stderr)
            continue

# ========================== Start Receiver Thread ==========================
recv_thread = threading.Thread(target=receive_data, daemon=True)
recv_thread.start()

# ========================== Main Processing Loop ==========================
print(f"✓ Connecting to friend at {config.FRIEND_IP}:{config.SERVER_PORT}")
print("✓ Starting video call...")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to read frame from camera", file=sys.stderr)
        break
    
    # Get frame dimensions for coordinate validation
    frame_height, frame_width = frame.shape[:2]
    
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run YOLO face detection
    results = model(frame)
    face_detected = False
    
    # Initialize with blank face (will be updated if face is detected)
    face_resized = np.zeros(config.FACE_SIZE, dtype=np.uint8)
    
    for result in results:
        for box in result.boxes:
            # Extract and validate bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = validate_coordinates(x1, y1, x2, y2, frame_height, frame_width)
            
            # Extract face region
            face = gray_frame[y1:y2, x1:x2]
            
            # Validate face region
            if not validate_face_region(face):
                continue
            
            face_detected = True
            
            # Resize and compress face
            face_resized = cv2.resize(face, config.FACE_SIZE)
            face_flatten = face_resized.flatten()
            compressed_face = compress_face_pca(face_flatten, eigenfaces, mean_face)
            
            # Send compressed face data
            try:
                face_data = serialize_data(compressed_face)
                server_socket.sendto(face_data, (config.FRIEND_IP, config.SERVER_PORT))
            except Exception as e:
                print(f"Warning: Failed to send data: {e}", file=sys.stderr)
    
    # Process received face with thread safety
    with data_lock:
        local_compressed_face = received_compressed_face
    
    if local_compressed_face is not None:
        reconstructed_face = reconstruct_face_pca(
            local_compressed_face,
            eigenfaces,
            mean_face,
            config.FACE_SIZE
        )
    else:
        reconstructed_face = np.zeros(config.FACE_SIZE, dtype=np.uint8)
    
    # Create display frame
    display_frame = np.zeros(
        (config.DISPLAY_SIZE[1], config.DISPLAY_SIZE[0], 3),
        dtype=np.uint8
    )
    
    # Prepare faces for display
    original_display = cv2.resize(face_resized, (config.BOX_SIZE, config.BOX_SIZE))
    reconstructed_display = cv2.resize(reconstructed_face, (config.BOX_SIZE, config.BOX_SIZE))

    # Convert grayscale to BGR for display
    original_display = cv2.cvtColor(original_display, cv2.COLOR_GRAY2BGR)
    reconstructed_display = cv2.cvtColor(reconstructed_display, cv2.COLOR_GRAY2BGR)

    # Add white borders
    border = 5
    original_display = cv2.copyMakeBorder(
        original_display, border, border, border, border,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    reconstructed_display = cv2.copyMakeBorder(
        reconstructed_display, border, border, border, border,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    # Calculate positions (accounting for border)
    display_size = config.BOX_SIZE + 2 * border
    left_y = 145
    left_x = 95
    right_x = 495
    
    # Place faces on display
    display_frame[left_y:left_y + display_size, left_x:left_x + display_size] = original_display
    display_frame[left_y:left_y + display_size, right_x:right_x + display_size] = reconstructed_display
    
    # Add labels
    cv2.putText(display_frame, config.YOUR_NAME, (160, 135),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, config.FRIEND_NAME, (560, 135),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show "No Face Detected" message if needed
    if not face_detected:
        cv2.putText(display_frame, "No Face Detected", (300, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow("PCA Video Call", display_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(config.WAIT_KEY_DELAY) & 0xFF == ord('q'):
        break

# ========================== Cleanup ==========================
cap.release()
cv2.destroyAllWindows()
server_socket.close()
print("\n✓ Video call ended successfully")

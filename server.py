import cv2
import numpy as np
import socket
import pickle
import threading
from ultralytics import YOLO

# Load YOLOv8 face detection model
model = YOLO("./yolov8n-face-lindevs.pt")  # Face detection model

# Load precomputed eigenfaces and mean face for grayscale images
eigenfaces = np.load("./eigen_faces.npy").astype(np.float32)[:, :1500]
mean_face = np.load("./mean_faces.npy")

# Network Config
IP = "192.168.228.158"  # Server IP
FRIEND_IP = "192.168.228.64"
PORT = 5142 

YOUR_NAME = "Shubham"
FRIEND_NAME = "Harsh"

BUFFER_SIZE = 65536

# Initialize socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((IP, PORT))
server_socket.setblocking(False)  # Non-blocking mode
print(f"Server listening on {IP}:{PORT}")

# Video Capture
cap = cv2.VideoCapture(0)

# Global variables for received data
received_compressed_face = None
received_addr = None

def receive_data():
    global received_compressed_face, received_addr
    while True:
        try:
            data, addr = server_socket.recvfrom(BUFFER_SIZE)
            received_compressed_face = pickle.loads(data)
            received_addr = addr
        except BlockingIOError:
            continue

# Start receiver thread
recv_thread = threading.Thread(target=receive_data, daemon=True)
recv_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = model(frame)  # Run YOLO on original frame
    face_detected = False
    face_resized = np.zeros((120, 120), dtype=np.uint8)
    
    for result in results:
        for box in result.boxes:
            face_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = gray_frame[y1:y2, x1:x2]  # Extract grayscale face region
            if face.size == 0:
                continue
            
            face_resized = cv2.resize(face, (120, 120))
            face_flatten = face_resized.flatten()
            compressed_face = eigenfaces.T @ (face_flatten - mean_face)
            
            face_data = pickle.dumps(compressed_face)
            server_socket.sendto(face_data, (FRIEND_IP, PORT))
    
    # Process received face
    if received_compressed_face is not None:
        reconstructed_face = eigenfaces @ received_compressed_face + mean_face
        reconstructed_face = np.clip(reconstructed_face, 0, 255).astype(np.uint8).reshape(120, 120)
    else:
        reconstructed_face = np.zeros((120, 120), dtype=np.uint8)
    
    # Display Both Faces
    display_frame = np.zeros((500, 800, 3), dtype=np.uint8)  # Black background
    original_display = cv2.resize(face_resized, (200, 200))  # Show original grayscale face
    reconstructed_display = cv2.resize(reconstructed_face, (200, 200))  # Show reconstructed grayscale face

    # Convert grayscale images to 3-channel grayscale for OpenCV display
    original_display = cv2.cvtColor(original_display, cv2.COLOR_GRAY2BGR)
    reconstructed_display = cv2.cvtColor(reconstructed_display, cv2.COLOR_GRAY2BGR)

    # Add white border to images
    border_thickness = 5
    original_display = cv2.copyMakeBorder(original_display, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    reconstructed_display = cv2.copyMakeBorder(reconstructed_display, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Set positions for display
    display_frame[145:355, 95:305] = original_display  # Adjusted for border
    display_frame[145:355, 495:705] = reconstructed_display  # Adjusted for border
    
    # Add labels above boxes
    cv2.putText(display_frame, YOUR_NAME, (160, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, FRIEND_NAME, (560, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show no face detected message if no face is found
    if not face_detected:
        cv2.putText(display_frame, "No Face Detected", (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    cv2.imshow("Grayscale Face PCA Video Call", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
server_socket.close()

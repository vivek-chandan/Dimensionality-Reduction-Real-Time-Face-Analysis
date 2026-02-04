"""
Real-time Face Detection and PCA Reconstruction Visualization

This script captures video from a webcam, detects faces using YOLOv8,
applies PCA compression/reconstruction, and displays the results side-by-side
with compression metrics.
"""

import cv2
import numpy as np
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
    reconstruct_face_pca,
    calculate_compression_stats
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

# ========================== Initialize Camera ==========================
try:
    cap = initialize_camera(config.CAMERA_INDEX)
except RuntimeError as e:
    print(f"Camera initialization failed: {e}", file=sys.stderr)
    sys.exit(1)

# ========================== Main Processing Loop ==========================
print("\n✓ Starting face PCA compression visualization...")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to read frame from camera", file=sys.stderr)
        break

    # Get frame dimensions for coordinate validation
    frame_height, frame_width = frame.shape[:2]

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create display background
    display_frame = np.full(
        (config.DISPLAY_SIZE[1], config.DISPLAY_SIZE[0], 3),
        config.BACKGROUND_COLOR,
        dtype=np.uint8
    )
    center_x = display_frame.shape[1] // 2
    center_y = display_frame.shape[0] // 2

    # Run YOLO face detection
    results = model(frame)

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
            
            # Calculate original face size for compression metrics
            original_h, original_w = face.shape[:2]
            original_face_size = original_h * original_w  # Grayscale: 1 byte per pixel

            # Resize face to standard PCA dimensions
            face_resized = cv2.resize(face, config.FACE_SIZE)

            # Flatten and compress using PCA
            face_vector = face_resized.flatten()
            compressed_representation = compress_face_pca(face_vector, eigenfaces, mean_face)

            # Reconstruct face from compressed representation
            reconstructed_face = reconstruct_face_pca(
                compressed_representation,
                eigenfaces,
                mean_face,
                config.FACE_SIZE
            )

            # Calculate reconstruction error (MSE)
            reconstruction_cost = np.mean((face_resized - reconstructed_face) ** 2)

            # Calculate compression statistics
            compressed_size = config.TOP_K_EIGENFACES * 4  # 4 bytes per float32
            compression_percentage, _ = calculate_compression_stats(
                original_face_size,
                compressed_size
            )

            # Prepare faces for display
            original_display = cv2.resize(face_resized, (config.BOX_SIZE, config.BOX_SIZE))
            reconstructed_display = cv2.resize(reconstructed_face, (config.BOX_SIZE, config.BOX_SIZE))

            # Convert grayscale to BGR for display
            original_display = cv2.cvtColor(original_display, cv2.COLOR_GRAY2BGR)
            reconstructed_display = cv2.cvtColor(reconstructed_display, cv2.COLOR_GRAY2BGR)

            # Calculate positions for display boxes
            left_box = (center_x - config.BOX_SIZE - 20, center_y - config.BOX_SIZE // 2)
            right_box = (center_x + 20, center_y - config.BOX_SIZE // 2)

            # Calculate safe boundaries for border drawing
            border = config.BORDER_THICKNESS
            
            # Ensure borders don't exceed frame boundaries
            left_y1 = max(0, left_box[1] - border)
            left_y2 = min(display_frame.shape[0], left_box[1] + config.BOX_SIZE + border)
            left_x1 = max(0, left_box[0] - border)
            left_x2 = min(display_frame.shape[1], left_box[0] + config.BOX_SIZE + border)
            
            right_y1 = max(0, right_box[1] - border)
            right_y2 = min(display_frame.shape[0], right_box[1] + config.BOX_SIZE + border)
            right_x1 = max(0, right_box[0] - border)
            right_x2 = min(display_frame.shape[1], right_box[0] + config.BOX_SIZE + border)

            # Draw colored borders around faces
            display_frame[left_y1:left_y2, left_x1:left_x2] = config.FRAME_COLOR
            display_frame[right_y1:right_y2, right_x1:right_x2] = config.FRAME_COLOR

            # Overlay the face images inside the frames
            display_frame[left_box[1]:left_box[1] + config.BOX_SIZE, 
                         left_box[0]:left_box[0] + config.BOX_SIZE] = original_display
            display_frame[right_box[1]:right_box[1] + config.BOX_SIZE, 
                         right_box[0]:right_box[0] + config.BOX_SIZE] = reconstructed_display

            # Draw white rectangle borders for clean frame effect
            cv2.rectangle(display_frame, 
                         (left_x1, left_y1), 
                         (left_x2, left_y2), 
                         (255, 255, 255), thickness=2)

            cv2.rectangle(display_frame, 
                         (right_x1, right_y1), 
                         (right_x2, right_y2), 
                         (255, 255, 255), thickness=2)

            # Add labels
            cv2.putText(display_frame, "Original", 
                       (left_box[0], left_box[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Reconstructed (MSE: {reconstruction_cost:.2f})", 
                       (right_box[0], right_box[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display compression percentage
            text_position = (center_x - 100, center_y + config.BOX_SIZE // 2 + 40)
            cv2.putText(display_frame, f"Compression: {compression_percentage:.2f}%", 
                       text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Face PCA Compression Visualization", display_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(config.WAIT_KEY_DELAY) & 0xFF == ord('q'):
        break

# ========================== Cleanup ==========================
cap.release()
cv2.destroyAllWindows()
print("\n✓ Visualization stopped successfully")

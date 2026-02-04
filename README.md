# Dimensionality Reduction & Real-Time Face Analysis

An implementation of PCA-based dimensionality reduction applied to real-time face detection and video communication. The project demonstrates how PCA compression can significantly reduce data size while maintaining visual quality, making it ideal for bandwidth-efficient video streaming.

## Features

- **Real-time Face Detection**: Uses YOLOv8 for accurate face detection
- **PCA Compression**: Reduces face data size by up to 97% using Principal Component Analysis
- **Live Visualization**: Side-by-side comparison of original and reconstructed faces
- **Video Call System**: Peer-to-peer video calling with PCA-compressed face transmission
- **Compression Metrics**: Real-time display of compression ratio and reconstruction error

## Project Structure

```
.
├── reconstruction_vis.py    # Face PCA compression visualization
├── server.py               # P2P video call with PCA compression
├── pca.ipynb              # PCA model training notebook
├── config.py              # Configuration settings
├── utils.py               # Utility functions and error handling
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Prerequisites

- Python 3.8 or higher
- Webcam
- YOLOv8 face detection model (`yolov8n-face-lindevs.pt`)
- Precomputed PCA data (`eigen_faces.npy`, `mean_faces.npy`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vivek-chandan/Dimensionality-Reduction-Real-Time-Face-Analysis.git
cd Dimensionality-Reduction-Real-Time-Face-Analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Generate PCA data:
   - Open and run `pca.ipynb` to generate `eigen_faces.npy` and `mean_faces.npy`
   - These files contain the precomputed eigenfaces and mean face for PCA

5. Download YOLOv8 face detection model:
   - Place the model file at `./yolov8n-face-lindevs.pt`

## Configuration

Edit `config.py` to customize:

- **PCA Settings**: Number of eigenfaces, face dimensions
- **Display Settings**: Window size, colors, borders
- **Network Settings**: IP addresses, ports (for video call)
- **Camera Settings**: Camera device index

Example:
```python
# config.py
TOP_K_EIGENFACES = 1000  # Number of eigenfaces to use
FACE_SIZE = (120, 120)   # Face dimensions for PCA
SERVER_IP = "192.168.1.100"  # Your IP address
FRIEND_IP = "192.168.1.101"  # Friend's IP address
```

## Usage

### Face PCA Visualization

Run the visualization to see real-time PCA compression/reconstruction:

```bash
python reconstruction_vis.py
```

- Shows original face and reconstructed face side-by-side
- Displays compression percentage and reconstruction error (MSE)
- Press 'q' to quit

### Video Call System

For peer-to-peer video calling with PCA compression:

1. **Configure Network Settings**:
   - Update `SERVER_IP` in `config.py` with your local IP
   - Update `FRIEND_IP` with your friend's IP
   - Ensure both computers are on the same network

2. **Run on Both Computers**:
```bash
python server.py
```

- Each user's face is compressed and sent to the other
- Received faces are reconstructed and displayed
- Press 'q' to quit

**Note**: Both users must run the script simultaneously with correct IP configurations.

## How It Works

### PCA Compression Process

1. **Face Detection**: YOLOv8 detects faces in each video frame
2. **Preprocessing**: Detected faces are resized to standard dimensions (120x120)
3. **Normalization**: Mean face is subtracted from the input face
4. **Projection**: Face is projected onto top K eigenfaces (PCA compression)
5. **Transmission**: Only K coefficients are transmitted (vs. 14,400 pixels)
6. **Reconstruction**: Received coefficients are used to reconstruct the face
7. **Display**: Original and reconstructed faces shown side-by-side

### Compression Efficiency

With default settings (1000 eigenfaces):
- **Original size**: 14,400 bytes (120×120 grayscale image)
- **Compressed size**: 4,000 bytes (1000 float32 coefficients)
- **Compression ratio**: ~72% reduction
- **Quality**: Minimal perceptual loss

## Code Improvements

This version includes several improvements over the original:

### Security & Safety
- ✅ Replaced `pickle` with `json` for network serialization (prevents code injection)
- ✅ Added comprehensive error handling for file operations
- ✅ Validated YOLO coordinates to prevent array index errors
- ✅ Thread-safe access to shared variables using locks

### Code Quality
- ✅ Centralized configuration in `config.py`
- ✅ Extracted utility functions to `utils.py`
- ✅ Added comprehensive docstrings
- ✅ Consistent code structure across both scripts
- ✅ Proper resource cleanup

### Performance
- ✅ Replaced busy-wait loop with socket timeout
- ✅ Removed redundant operations
- ✅ Added boundary checks for display operations

### Documentation
- ✅ Enhanced README with setup instructions
- ✅ Added inline comments explaining PCA process
- ✅ Created `requirements.txt` for easy dependency management
- ✅ Added `.gitignore` for Python projects

## Troubleshooting

### Camera Issues
- Ensure camera is not being used by another application
- Try changing `CAMERA_INDEX` in `config.py` (usually 0, 1, or 2)

### File Not Found Errors
- Run `pca.ipynb` to generate required `.npy` files
- Ensure YOLOv8 model file is in the correct location

### Network Issues (Video Call)
- Verify both computers are on the same network
- Check firewall settings (allow UDP on port 5142)
- Ensure IP addresses in `config.py` are correct
- Try pinging the other computer to verify connectivity

### Performance Issues
- Reduce `TOP_K_EIGENFACES` for faster processing
- Ensure adequate lighting for better face detection
- Close other applications to free up resources

## Technical Details

- **Face Detection**: YOLOv8n (lightweight, real-time)
- **Dimensionality Reduction**: PCA (Principal Component Analysis)
- **Image Processing**: OpenCV
- **Network Protocol**: UDP (User Datagram Protocol)
- **Data Format**: JSON (for safe serialization)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV for computer vision operations
- PCA algorithm for dimensionality reduction


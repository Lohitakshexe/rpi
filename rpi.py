import cv2
import time
from ultralytics import YOLO
import socket
import pickle
import struct
import os
import json
import argparse
from datetime import datetime

# Load the YOLOv8m model
print("Loading model...")
model = YOLO("yolov8m.pt")

# Camera configuration
DEFAULT_IP_CAMERA_URL = "http://192.168.1.100:8080/video"  # Default IP camera URL
DEFAULT_WEBCAM_ID = 0  # Default built-in webcam ID (usually 0)

# Detection tracking variables
detection_count = 0
detection_history = []
detection_log_file = "detection_log.json"
detection_stats_file = "detection_stats.csv"

# Create detection logs directory if it doesn't exist
log_dir = "detection_logs"
os.makedirs(log_dir, exist_ok=True)

# Function to save detection data
def save_detection_data(detection_data, count):
    # Save to JSON log file
    timestamp = datetime.fromtimestamp(detection_data['timestamp']).strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"detection_{timestamp}.json")
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {
        'timestamp': detection_data['timestamp'],
        'count': count,
        'names': detection_data['names'],
        'boxes': detection_data['boxes'].tolist() if hasattr(detection_data['boxes'], 'tolist') else detection_data['boxes']
    }
    
    with open(log_file, 'w') as f:
        json.dump(serializable_data, f)
    
    # Append to stats CSV file
    stats_file = os.path.join(log_dir, detection_stats_file)
    file_exists = os.path.isfile(stats_file)
    
    with open(stats_file, 'a') as f:
        if not file_exists:
            f.write("timestamp,active_detections\n")
        f.write(f"{timestamp},{count}\n")
    
    print(f"Saved detection data: {count} active detections")

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection with Camera Options')
    parser.add_argument('--camera-type', type=str, choices=['ip', 'webcam'], default='ip',
                        help='Camera type: "ip" for IP camera or "webcam" for built-in webcam')
    parser.add_argument('--ip-camera-url', type=str, default=DEFAULT_IP_CAMERA_URL,
                        help='URL for the IP camera stream')
    parser.add_argument('--webcam-id', type=int, default=DEFAULT_WEBCAM_ID,
                        help='Device ID for the webcam (usually 0 for built-in)')
    return parser.parse_args()

# Get camera configuration
args = parse_arguments()

# Open video stream based on selected camera type
if args.camera_type == 'ip':
    print(f"Connecting to IP camera at {args.ip_camera_url}...")
    cap = cv2.VideoCapture(args.ip_camera_url)
    camera_label = "IP Camera"
else:  # webcam
    print(f"Connecting to webcam (ID: {args.webcam_id})...")
    cap = cv2.VideoCapture(args.webcam_id)
    camera_label = "Webcam"

if not cap.isOpened():
    print(f"ERROR: Cannot open {args.camera_type} stream.")
    exit()

# Configure the streaming server details (if needed)
STREAM_ENABLED = False  # Set to True if you want to stream to another server
if STREAM_ENABLED:
    STREAM_HOST = '192.168.1.101'  # Replace with receiver IP
    STREAM_PORT = 9999
    
    # Create a socket connection
    try:
        stream_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        stream_socket.connect((STREAM_HOST, STREAM_PORT))
        print(f"Connected to streaming server at {STREAM_HOST}:{STREAM_PORT}")
    except Exception as e:
        print(f"WARNING: Could not connect to streaming server: {e}")
        STREAM_ENABLED = False

print("Starting detection loop. Press 'q' to exit.")

# Variables for periodic saving
last_save_time = time.time()
save_interval = 5  # Save every 5 seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to grab frame.")
            continue

        # Run YOLOv8 prediction
        results = model.predict(frame, conf=0.3, verbose=False)
        
        # Count active detections (number of detected objects)
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            detection_count = len(results[0].boxes)
        else:
            detection_count = 0
        
        # Get detection data
        current_time = time.time()
        detection_data = {
            'boxes': results[0].boxes.data.cpu().numpy() if detection_count > 0 else [],
            'names': results[0].names,
            'timestamp': current_time
        }
        
        # Add to history
        detection_history.append({
            'timestamp': current_time,
            'count': detection_count
        })
        
        # Limit history size to prevent memory issues
        if len(detection_history) > 1000:
            detection_history = detection_history[-1000:]
        
        # Save detection data periodically
        if current_time - last_save_time >= save_interval and detection_count > 0:
            save_detection_data(detection_data, detection_count)
            last_save_time = current_time
        
        # Stream data if enabled
        if STREAM_ENABLED:
            try:
                # Serialize the detection data
                data = pickle.dumps(detection_data)
                
                # Pack the message size and send it
                message_size = struct.pack("L", len(data))
                stream_socket.sendall(message_size + data)
            except Exception as e:
                print(f"ERROR: Failed to send data: {e}")
                STREAM_ENABLED = False

        # Plot the detection results with count
        annotated_frame = results[0].plot()
        
        # Add detection count to the frame
        cv2.putText(annotated_frame, f"Detections: {detection_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame in a window
        cv2.imshow(f"YOLOv8m {camera_label} Feed", annotated_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Cleanup
    print("Cleaning up resources...")
    if STREAM_ENABLED:
        stream_socket.close()
    cap.release()
    cv2.destroyAllWindows()
    
    # Save final statistics
    print(f"Total detection sessions recorded: {len(detection_history)}")
    print(f"Detection logs saved in: {log_dir}/")

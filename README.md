# YOLOv8 Real-Time Object Detection

This project demonstrates real-time object detection using the YOLOv8 model and a webcam feed. The bounding boxes and labels for detected objects are displayed on the video feed.

## Prerequisites

Ensure you have the following installed:

1. Python (>=3.8)
2. Virtual environment (optional but recommended)
3. Required Python packages:
    - `ultralytics`
    - `opencv-python`
    - `torch`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repository/yolo-real-time.git
   cd yolo-real-time
   ```

2. **Create and Activate a Virtual Environment (Optional)**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install ultralytics opencv-python torch
   ```

4. **Download YOLOv8 Pretrained Weights**:
   Download `yolov8n.pt` (YOLOv8 Nano model) from the [Ultralytics website](https://github.com/ultralytics/ultralytics).
   Place the file in the project directory.

## Usage

1. Save the following Python script as `yolo.py`:

   ```python
   import cv2
   from ultralytics import YOLO

   # Load the YOLOv8 model
   model = YOLO('yolov8n.pt')

   # Start webcam feed
   cap = cv2.VideoCapture(0)

   while True:
       ret, frame = cap.read()
       if not ret:
           break

       # Perform object detection
       results = model(frame)

       # Check if there are any detections
       if results:
           for result in results:
               if result.boxes:
                   # Loop through each detection and draw the bounding box
                   for i, box in enumerate(result.boxes.xyxy):  # xyxy are the bounding box coordinates
                       x1, y1, x2, y2 = map(int, box)
                       cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box

                       # Display the label (if available) on the frame
                       if result.names:
                           cls_id = int(result.boxes.cls[i])  # Convert the class ID to an integer
                           label = result.names.get(cls_id, "Unknown")  # Safely get the label
                           cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

       # Show the frame with bounding boxes
       cv2.imshow("Object Detection", frame)

       # Exit when 'q' is pressed
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   # Release the webcam and close OpenCV windows
   cap.release()
   cv2.destroyAllWindows()
   ```

2. **Run the Script**:
   ```bash
   python yolo.py
   ```

3. **View the Output**:
   - The webcam feed will appear in a window labeled "Object Detection."
   - Detected objects will have bounding boxes and labels displayed in real time.

4. **Exit**:
   - Press `q` to close the video feed and exit the application.

## Notes

- Ensure your webcam is connected and accessible.
- You can replace `yolov8n.pt` with other YOLOv8 model weights (e.g., `yolov8s.pt`, `yolov8m.pt`) for different performance and speed trade-offs.
- If you encounter any issues, ensure all dependencies are correctly installed and the webcam is functioning.

## Troubleshooting

- **No video feed**:
  Ensure your webcam is not being used by another application.

- **Model not found**:
  Verify that `yolov8n.pt` is in the project directory.

- **Performance issues**:
  Use a lower-resolution webcam feed for better speed or switch to a faster model like `yolov8n.pt`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.


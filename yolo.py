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

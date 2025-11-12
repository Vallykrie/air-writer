import cv2
from ultralytics import YOLO

# 1. Load your CUSTOM trained model
model = YOLO('runs/pose/train4/weights/best.pt')

# 2. Open a connection to the webcam (source 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam feed. Press 'q' to quit.")

# 3. Loop through frames from the webcam
while True:
    # Read a new frame
    success, frame = cap.read()

    if not success:
        print("Error: Could not read frame.")
        break

    # 4. Run YOLOv8 pose estimation on the frame
    # Setting stream=True is more efficient for video feeds
    results = model(frame, stream=True)

    # 5. Process the results and display
    # 'results' is a generator. Loop through it.
    for r in results:
        # 'plot()' returns a frame with the detections and keypoints drawn on it
        annotated_frame = r.plot()

        # You can also access the keypoints data directly if needed
        # keypoints = r.keypoints  # Get keypoints object
        # print(keypoints.xy)      # Print keypoint coordinates

        # Display the annotated frame
        cv2.imshow("YOLOv11 Hand Pose Estimation", annotated_frame)

    # 6. Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Release resources
cap.release()
cv2.destroyAllWindows()
print("Webcam feed stopped.")
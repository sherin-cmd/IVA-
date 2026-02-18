import cv2
from ultralytics import YOLO

# ===============================
# Load YOLO Model
# ===============================
model = YOLO("yolov8n.pt")   # Pretrained lightweight model


# ===============================
# IMAGE OBJECT DETECTION
# ===============================
def detect_image(image_path):
    results = model(image_path)

    # Plot results
    annotated_frame = results[0].plot()

    cv2.imshow("Object Detection - Image", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===============================
# VIDEO OBJECT DETECTION
# ===============================
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("Object Detection - Video", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    detect_image("image.jpg")   # Replace with your image
    detect_video("video.mp4")   # Replace with your video
![1000339823](https://github.com/user-attachments/assets/2c62f278-e286-4409-aed6-36359c9523e2)

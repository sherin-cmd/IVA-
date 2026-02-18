import cv2
import numpy as np

# ===============================
# IMAGE GEOMETRIC TRANSFORMS
# ===============================
def image_transforms(image_path):
    img = cv2.imread(image_path)
    rows, cols = img.shape[:2]

    # 1️⃣ Translation
    M_translate = np.float32([[1, 0, 50],
                              [0, 1, 30]])
    translated = cv2.warpAffine(img, M_translate, (cols, rows))

    # 2️⃣ Rotation
    M_rotate = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    rotated = cv2.warpAffine(img, M_rotate, (cols, rows))

    # 3️⃣ Scaling
    scaled = cv2.resize(img, None, fx=1.5, fy=1.5)

    # 4️⃣ Shearing
    M_shear = np.float32([[1, 0.5, 0],
                          [0, 1, 0]])
    sheared = cv2.warpAffine(img, M_shear, (int(cols*1.5), rows))

    # 5️⃣ Affine Transform
    pts1 = np.float32([[50,50], [200,50], [50,200]])
    pts2 = np.float32([[10,100], [200,50], [100,250]])
    M_affine = cv2.getAffineTransform(pts1, pts2)
    affine = cv2.warpAffine(img, M_affine, (cols, rows))

    # 6️⃣ Perspective Transform
    pts1 = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
    pts2 = np.float32([[0,0], [cols-1,50], [50,rows-1], [cols-50,rows-50]])
    M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
    perspective = cv2.warpPerspective(img, M_perspective, (cols, rows))

    # Show results
    cv2.imshow("Original", img)
    cv2.imshow("Translated", translated)
    cv2.imshow("Rotated", rotated)
    cv2.imshow("Scaled", scaled)
    cv2.imshow("Sheared", sheared)
    cv2.imshow("Affine", affine)
    cv2.imshow("Perspective", perspective)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===============================
# VIDEO GEOMETRIC TRANSFORM
# ===============================
def video_transform(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rows, cols = frame.shape[:2]

        # Rotate each frame
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
        transformed = cv2.warpAffine(frame, M, (cols, rows))

        cv2.imshow("Original Video", frame)
        cv2.imshow("Rotated Video", transformed)

        if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    image_transforms("image.jpg")   # Replace with your image
    video_transform("video.mp4")    # Replace with your video
![1000339824](https://github.com/user-attachments/assets/85d061b7-6030-4e7f-850d-6284dd6adf54)

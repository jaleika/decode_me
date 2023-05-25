import cv2
from matplotlib import pyplot as plt
import cvlib as cv
from cvlib.object_detection import detect_common_objects, draw_bbox

# Load and process images
image_paths = [
    "face_detection/test_image_1.jpg",
    "face_detection/test_image_2.jpg",
    "face_detection/test_image_3.jpg",
]
images = [cv2.imread(image_path) for image_path in image_paths]

# Plot the images with bounding boxes
fig, axs = plt.subplots(1, len(images), figsize=(12, 6))
for i, image in enumerate(images):
    # Perform face detection
    faces, confidences = cv.detect_face(image)

    # Draw bounding boxes around the detected faces (if any)
    for (x, y, w, h), confidence in zip(faces, confidences):
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

    axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[i].set_title(f"Image {i+1}")
    axs[i].axis("off")

plt.tight_layout()
plt.show()

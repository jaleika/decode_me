from pathlib import Path

import cv2
import cvlib as cv

# from matplotlib import pyplot as plt
# from cvlib.object_detection import detect_common_objects, draw_bbox

# Load and process images
# image_paths = [
#     "face_detection/test_image_1.jpg",
#     "face_detection/test_image_2.jpg",
#     "face_detection/test_image_3.jpg",
# ]
# images = [cv2.imread(image_path) for image_path in image_paths]

# # Plot the images with bounding boxes
# fig, axs = plt.subplots(1, len(images), figsize=(12, 6))
# for i, image in enumerate(images):
#     # Perform face detection
#     faces, confidences = cv.detect_face(image)

#     # Draw bounding boxes around the detected faces (if any)
#     for (x, y, w, h), confidence in zip(faces, confidences):
#         cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

#     axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     axs[i].set_title(f"Image {i+1}")
#     axs[i].axis("off")

# plt.tight_layout()
# plt.show()


def crop_faces(image_path: str = "face_detection/test_image_1.jpg"):
    image = cv2.imread(image_path)
    # Perform face detection
    faces, _confidences = cv.detect_face(image)
    for index, face in enumerate(faces):
        cropped_image = image[
            face[1] : face[3],
            face[0] : face[2],
        ]
        path = Path(image_path)
        coordinates_string = ",".join(map(str, face))
        cv2.imwrite(
            f"face_detection/export/{path.stem}_-_{coordinates_string}.jpg",
            cropped_image,
        )


if __name__ == "__main__":
    crop_faces()

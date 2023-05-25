from pathlib import Path
import cv2
import cvlib as cv


def crop_faces(image_path: str = "face_detection/test_image_5.jpg"):

    """Extracts faces from an image and save them as separate images in "export" folder
    Args: image path
    """
    # Load an iamge from the path
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

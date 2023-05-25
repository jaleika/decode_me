from yolo5face.get_model import get_model
from pathlib import Path
import cv2


def crop_faces(image_path: str = "face_detection/test_image_1.jpg"):

    """Extracts faces from an image and save them as separate images in "export" folder
    Args: image path
    """
    # Load the model
    model = get_model("yolov5n", gpu=-1, target_size=512, min_face=24)
    # Load an iamge from the path
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection
    faces, _key_points = model(image)
    for index, face in enumerate(faces[0]):
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
